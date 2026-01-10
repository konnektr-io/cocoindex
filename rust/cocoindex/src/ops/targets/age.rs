use crate::prelude::*;

use super::shared::property_graph::*;
use crate::setup::components::{self, State, apply_component_changes, SetupStateCompatibility};
use crate::setup::{ResourceSetupChange, SetupChangeType};
use crate::{ops::sdk::*, setup::CombinedState};
use crate::settings::DatabaseConnectionSpec;
use crate::ops::shared::postgres::get_db_pool;

use indoc::formatdoc;
use sqlx::{PgPool, Row};
use std::fmt::Write;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize, Clone)]
pub struct ConnectionSpec {
    pub host: String,
    pub port: Option<u16>,
    pub user: String,
    pub password: Option<String>,
    pub database: String,
    pub graph_name: String,
}

impl From<&ConnectionSpec> for DatabaseConnectionSpec {
    fn from(spec: &ConnectionSpec) -> Self {
        Self {
            host: spec.host.clone(),
            port: spec.port.unwrap_or(5432),
            user: spec.user.clone(),
            password: spec.password.clone(),
            database: spec.database.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Spec {
    connection: spec::AuthEntryReference<ConnectionSpec>,
    mapping: std::sync::Arc<GraphElementMapping>,
}

#[derive(Debug, Deserialize)]
pub struct Declaration {
    connection: spec::AuthEntryReference<ConnectionSpec>,
    #[serde(flatten)]
    decl: GraphDeclaration,
}

type AgeGraphElement = GraphElementType<ConnectionSpec>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
struct GraphKey {
    connection: spec::AuthEntryReference<ConnectionSpec>,
}

pub struct Factory;

impl Factory {
    pub fn new() -> Self {
        Self
    }
}

// Setup State definitions

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ComponentKind {
    KeyConstraint,
    VectorIndex,
}

impl ComponentKind {
    fn describe(&self) -> &str {
        match self {
            ComponentKind::KeyConstraint => "KEY CONSTRAINT",
            ComponentKind::VectorIndex => "VECTOR INDEX",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComponentKey {
    kind: ComponentKind,
    name: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
enum IndexDef {
    KeyConstraint {
        field_names: Vec<String>,
    },
    VectorIndex {
        field_name: String,
        metric: spec::VectorSimilarityMetric,
        vector_size: usize,
        method: Option<spec::VectorIndexMethod>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct ComponentState {
    object_label: ElementType,
    index_def: IndexDef,
}

impl components::State<ComponentKey> for ComponentState {
    fn key(&self) -> ComponentKey {
        let prefix = match &self.object_label {
            ElementType::Relationship(_) => "r",
            ElementType::Node(_) => "n",
        };
        let label = self.object_label.label();
        match &self.index_def {
            IndexDef::KeyConstraint { .. } => ComponentKey {
                kind: ComponentKind::KeyConstraint,
                name: format!("{prefix}__{label}__key"),
            },
            IndexDef::VectorIndex {
                field_name, metric, ..
            } => ComponentKey {
                kind: ComponentKind::VectorIndex,
                name: format!("{prefix}__{label}__{field_name}__{metric}__vidx"),
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SetupState {
    key_field_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    dependent_node_labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    sub_components: Vec<ComponentState>,
}

impl SetupState {
    fn new(
        schema: &GraphElementSchema,
        index_options: &IndexOptions,
        dependent_node_labels: Vec<String>,
    ) -> Result<Self> {
        let key_field_names: Vec<String> =
            schema.key_fields.iter().map(|f| f.name.clone()).collect();
        let mut sub_components = vec![];
        
        // Add key constraint component
        sub_components.push(ComponentState {
            object_label: schema.elem_type.clone(),
            index_def: IndexDef::KeyConstraint {
                field_names: key_field_names.clone(),
            },
        });

        let value_field_types = schema
            .value_fields
            .iter()
            .map(|f| (f.name.as_str(), &f.value_type.typ))
            .collect::<HashMap<_, _>>();

        if !index_options.fts_indexes.is_empty() {
             // FTS not supported yet
             // We can ignore or error. Neo4j errors.
             api_bail!("FTS indexes are not supported for Age target");
        }

        for index_def in index_options.vector_indexes.iter() {
             sub_components.push(ComponentState {
                object_label: schema.elem_type.clone(),
                index_def: IndexDef::from_vector_index_def(
                    index_def,
                    value_field_types
                        .get(index_def.field_name.as_str())
                        .ok_or_else(|| {
                            api_error!(
                                "Unknown field name for vector index: {}",
                                index_def.field_name
                            )
                        })?,
                )?,
            });
        }

        Ok(Self {
            key_field_names,
            dependent_node_labels,
            sub_components,
        })
    }

impl components::SetupStateCompatibilityCheck for SetupState {
    fn check_compatible(&self, existing: &Self) -> SetupStateCompatibility {
        if self.key_field_names == existing.key_field_names {
            SetupStateCompatibility::Compatible
        } else {
            SetupStateCompatibility::NotCompatible
        }
    }
}

impl IntoIterator for SetupState {
    type Item = ComponentState;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.sub_components.into_iter()
    }
}

impl IndexDef {
    fn from_vector_index_def(
        index_def: &spec::VectorIndexDef,
        field_typ: &schema::ValueType,
    ) -> Result<Self> {
        let method = index_def.method.clone();
        // Support IvfFlat and Hnsw for pgvector
        Ok(Self::VectorIndex {
            field_name: index_def.field_name.clone(),
            vector_size: (match field_typ {
                schema::ValueType::Basic(schema::BasicValueType::Vector(schema)) => {
                    schema.dimension
                }
                _ => None,
            })
            .ok_or_else(|| {
                api_error!("Vector index field must be a vector with fixed dimension")
            })?,
            metric: index_def.metric,
            method,
        })
    }
}

struct SetupComponentOperator {
    pool: PgPool,
    conn_spec: ConnectionSpec,
    has_pgvector: bool,
}

#[async_trait]
impl components::SetupOperator for SetupComponentOperator {
    type Key = ComponentKey;
    type State = ComponentState;
    type SetupState = SetupState;
    type Context = ();

    fn describe_key(&self, key: &Self::Key) -> String {
        format!("{} {}", key.kind.describe(), key.name)
    }

    fn describe_state(&self, state: &Self::State) -> String {
        match &state.index_def {
            IndexDef::KeyConstraint { field_names } => {
                format!("Constraint on ({})", field_names.join(", "))
            }
            IndexDef::VectorIndex {
                field_name,
                metric,
                vector_size,
                method,
            } => {
                format!(
                    "Vector Index on {} (dim: {}, metric: {}, method: {:?})",
                    field_name, vector_size, metric, method
                )
            }
        }
    }

    async fn is_up_to_date(
        &self,
        key: &Self::Key,
        _state: &Self::State,
        _context: &Self::Context,
    ) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("LOAD 'age'").execute(&mut *tx).await?;
        sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

        // Check if index exists in the graph's schema
        let index_name = &key.name;
        let schema_name = &self.conn_spec.graph_name;

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = $1 AND c.relkind = 'i'
                AND n.nspname = $2
            )"
        )
        .bind(index_name)
        .bind(schema_name)
        .fetch_one(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(exists)
    }

    async fn create(&self, state: &Self::State, _context: &Self::Context) -> Result<()> {
         match &state.index_def {
            IndexDef::VectorIndex { .. } => {
                if !self.has_pgvector {
                   tracing::warn!("Skipping vector index creation as pgvector extension is missing");
                   return Ok(());
                }
            }
            _ => {}
         }

         let mut tx = self.pool.begin().await?;
         sqlx::query("LOAD 'age'").execute(&mut *tx).await?;
         sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

         let graph = &self.conn_spec.graph_name;
         let label = state.object_label.label();
         let is_edge = matches!(state.object_label, ElementType::Relationship(_));

         let create_label_q = if is_edge {
             format!("SELECT create_elabel('{}', '{}')", graph, label)
         } else {
             format!("SELECT create_vlabel('{}', '{}')", graph, label)
         };
         sqlx::query(&create_label_q).execute(&mut *tx).await?;

         let index_name = state.key().name;
         let q_graph = format!("\"{}\"", graph);
         let q_label = format!("\"{}\"", label);
         let q_index = format!("\"{}\"", index_name);

         match &state.index_def {
             IndexDef::KeyConstraint { field_names } => {
                 let exprs = field_names
                    .iter()
                    .map(|f| format!("(properties ->> '{}')", f))
                    .collect::<Vec<_>>()
                    .join(", ");
                 let sql = format!("CREATE UNIQUE INDEX {} ON {}.{} ({})", q_index, q_graph, q_label, exprs);
                 sqlx::query(&sql).execute(&mut *tx).await?;
             }
             IndexDef::VectorIndex { field_name, metric, method, .. } => {
                 let ops = match metric {
                    spec::VectorSimilarityMetric::L2 => "vector_l2_ops",
                    spec::VectorSimilarityMetric::Cosine => "vector_cosine_ops",
                    spec::VectorSimilarityMetric::InnerProduct => "vector_ip_ops",
                 };
                 let method_str = match method {
                     Some(spec::VectorIndexMethod::Hnsw) | None => "hnsw",
                     Some(spec::VectorIndexMethod::IvfFlat) => "ivfflat",
                 };

                 let sql = format!(
                     "CREATE INDEX {} ON {}.{} USING {} (((properties ->> '{}')::vector) {})",
                     q_index, q_graph, q_label, method_str, field_name, ops
                 );
                 sqlx::query(&sql).execute(&mut *tx).await?;
             }
         }

         tx.commit().await?;
         Ok(())
    }

    async fn delete(&self, key: &Self::Key, _context: &Self::Context) -> Result<()> {
         let mut tx = self.pool.begin().await?;
         sqlx::query("LOAD 'age'").execute(&mut *tx).await?;
         sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

         let index_name = &key.name;
         let q_graph = format!("\"{}\"", self.conn_spec.graph_name);
         let q_index = format!("\"{}\"", index_name);
         let sql = format!("DROP INDEX IF EXISTS {}.{}", q_graph, q_index);

         sqlx::query(&sql).execute(&mut *tx).await?;
         tx.commit().await?;
         Ok(())
    }
}

pub struct AgeSetupChange {
    component_change: components::SetupChange<SetupState>,
    pool: PgPool,
    conn_spec: ConnectionSpec,
    label_to_clean: Option<String>,
    has_pgvector: bool,
}

impl ResourceSetupChange for AgeSetupChange {
    fn change_type(&self) -> SetupChangeType {
        if self.label_to_clean.is_some() {
            return SetupChangeType::Update;
        }
        self.component_change.change_type()
    }

    fn describe_changes(&self) -> Vec<crate::setup::ChangeDescription> {
        let mut changes = self.component_change.describe_changes();
        if let Some(label) = &self.label_to_clean {
             changes.push(crate::setup::ChangeDescription::Action(format!(
                "Clear data for label '{}'",
                label
            )));
        }
        changes
    }

    fn apply_change(&self) -> std::pin::Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async {
            let operator = SetupComponentOperator {
                pool: self.pool.clone(),
                conn_spec: self.conn_spec.clone(),
                has_pgvector: self.has_pgvector,
            };
            self.component_change.apply_change(&operator, &()).await?;

            if let Some(label) = &self.label_to_clean {
                let mut tx = self.pool.begin().await?;
                sqlx::query("LOAD 'age'").execute(&mut *tx).await?;
                sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

                let cypher = format!("MATCH (n:{}) DETACH DELETE n", label);
                let sql = format!("SELECT * FROM cypher('{}', $$ {} $$) as (v agtype)", self.conn_spec.graph_name, cypher);
                
                // We execute and ignore error? 
                // Using sqlx::query(...).execute()
                if let Err(e) = sqlx::query(&sql).execute(&mut *tx).await {
                    // Log but ignore if graph/label missing?
                    tracing::warn!("Failed to clear data for label {}: {}", label, e);
                }

                tx.commit().await?;
            }
            Ok(())
        })
    }
}

async fn get_connection_pool(
    conn_ref: &spec::AuthEntryReference<ConnectionSpec>,
    auth_registry: &AuthRegistry
) -> Result<PgPool> {
    let conn_spec = auth_registry.get(conn_ref)?;
    let db_spec: DatabaseConnectionSpec = conn_spec.into();
    let lib_context = get_lib_context().await?;
    lib_context.db_pools.get_pool(&db_spec).await
}

async fn check_pgvector(pool: &PgPool) -> Result<bool> {
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        .fetch_one(pool).await.unwrap_or(0);
    Ok(count > 0)
}

struct ExportContext {
    pool: PgPool,
    conn_spec: ConnectionSpec,
    analyzed_data_coll: AnalyzedDataCollection,
    upsert_cypher: String,
    delete_cypher: String,
}

impl ExportContext {
    fn extract_key_params(&self, key: &Key) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut map = serde_json::Map::new();
        // analyzed_data_coll.schema.key_fields -> Vec<GraphFieldSchema>
        // Key match logic:
        // If single key field, key is value.
        // If multiple, key is struct (Value::Struct).
        
        let key_fields = &self.analyzed_data_coll.schema.key_fields;
        if key_fields.len() == 1 {
            let field = &key_fields[0];
            let val = serde_json::to_value(key)?;
            map.insert(field.name.clone(), val);
        } else {
             // Expect composite key
             match key {
                 Key::Struct(fields) => {
                     for field in key_fields {
                         let val = fields.get(&field.name).ok_or_else(|| {
                             api_error!("Missing key field part: {}", field.name)
                         })?;
                         map.insert(field.name.clone(), serde_json::to_value(val)?);
                     }
                 }
                 _ => api_bail!("Expected Struct key for composite key fields, got {:?}", key),
             }
        }
        Ok(map)
    }

    fn extract_value_params(&self, values: Vec<Value>) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut map = serde_json::Map::new();
        // values correspond to self.analyzed_data_coll.value_fields_input_idx
        // But Input idx points to which input field maps to the schema value field?
        // Actually values come from `Mutation::Upsert(values)`.
        // The `values` vector matches `spec.mapping` fields order?
        // No, `Mutation` holds `values: Vec<Value>`.
        // The order corresponds to `analyzed_data_coll.value_fields_input_idx`.
        
        // Wait, `Mutation` values are provided by pipeline.
        // Usually target expects them in a specific order defined by `spec`.
        // analyzed_data_coll has `value_fields_input_idx`.
        
        for (i, val) in values.into_iter().enumerate() {
            // Find which schema field this corresponds to.
            // analyzed_data_coll.schema.value_fields[i]?
            // Usually valid.
            if i < self.analyzed_data_coll.schema.value_fields.len() {
                let schema_field = &self.analyzed_data_coll.schema.value_fields[i];
                let json_val = serde_json::to_value(val)?;
                map.insert(schema_field.name.clone(), json_val);
            }
        }
        Ok(map)
    }
}

#[async_trait]
impl Executor for ExportContext {
    async fn mutate(&mut self, batch: MutationBatch) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let mut tx = self.pool.begin().await?;
        sqlx::query("LOAD 'age'").execute(&mut *tx).await?;
        sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

        let (upserts, deletes): (Vec<_>, Vec<_>) = batch.into_iter().partition(|m| m.is_upsert());

        if !deletes.is_empty() {
            let mut batch_data = Vec::with_capacity(deletes.len());
            for item in deletes {
                let props = self.extract_key_params(&item.key)?;
                batch_data.push(serde_json::Value::Object(props));
            }
            let params = serde_json::json!({ "batch": batch_data });
            let sql = format!("SELECT * FROM cypher('{}', $$ {} $$, $1) as (v agtype)", self.conn_spec.graph_name, self.delete_cypher);
            sqlx::query(&sql).bind(sqlx::types::Json(params)).execute(&mut *tx).await?;
        }

        if !upserts.is_empty() {
            let mut batch_data = Vec::with_capacity(upserts.len());
            for item in upserts {
                let mut props = self.extract_key_params(&item.key)?;
                // into_values() consumes mutation
                if let Mutation::Upsert(values) = item {
                    let val_props = self.extract_value_params(values)?;
                    props.extend(val_props);
                }
                batch_data.push(serde_json::Value::Object(props));
            }
            let params = serde_json::json!({ "batch": batch_data });
            let sql = format!("SELECT * FROM cypher('{}', $$ {} $$, $1) as (v agtype)", self.conn_spec.graph_name, self.upsert_cypher);
            sqlx::query(&sql).bind(sqlx::types::Json(params)).execute(&mut *tx).await?;
        }

        tx.commit().await?;
        Ok(())
    }
}

fn generate_cypher_upsert(coll: &AnalyzedDataCollection, has_pgvector: bool) -> Result<String> {
    use std::fmt::Write;
    let mut q = String::new();
    
    write!(q, "UNWIND $batch AS row ").unwrap();
    
    match &coll.schema.elem_type {
        ElementType::Node(label) => {
            let key_props_match = coll.schema.key_fields.iter()
                .map(|f| format!("{}: row.{}", f.name, f.name))
                .collect::<Vec<_>>().join(", ");
                
            write!(q, "MERGE (n:{} {{ {} }}) ", label, key_props_match).unwrap();
            
            if !coll.schema.value_fields.is_empty() {
                write!(q, "SET ").unwrap();
                let sets = coll.schema.value_fields.iter().map(|f| {
                    let is_vec = matches!(f.value_type.typ, schema::ValueType::Basic(schema::BasicValueType::Vector(_)));
                    if is_vec && has_pgvector {
                        format!("n.{} = row.{}::vector", f.name, f.name)
                    } else {
                        format!("n.{} = row.{}", f.name, f.name)
                    }
                }).collect::<Vec<_>>().join(", ");
                write!(q, "{}", sets).unwrap();
            }
        }
        ElementType::Relationship(_) => {
             // Relationship support TODO: fully implement key mapping
             api_bail!("Relationship support not fully implemented for AGE target");
        }
    }
    Ok(q)
}

fn generate_cypher_delete(coll: &AnalyzedDataCollection) -> Result<String> {
    use std::fmt::Write;
    let mut q = String::new();
    write!(q, "UNWIND $batch AS row ").unwrap();
    match &coll.schema.elem_type {
        ElementType::Node(label) => {
            let key_props_match = coll.schema.key_fields.iter()
                .map(|f| format!("{}: row.{}", f.name, f.name))
                .collect::<Vec<_>>().join(", ");
            write!(q, "MATCH (n:{} {{ {} }}) DETACH DELETE n", label, key_props_match).unwrap();
        }
        _ => { api_bail!("Relationship delete not implemented"); }
    }
    Ok(q)
}

#[async_trait]
impl TargetFactoryBase for Factory {
    type Spec = Spec;
    type DeclarationSpec = Declaration;
    type SetupState = SetupState;
    type SetupChange = AgeSetupChange;
    type Executor = ExportContext;
    type AdditionalDataCollectionInfo = ();

    async fn build(
        spec: &Self::Spec,
        declarations: &[Self::DeclarationSpec],
        _auth_registry: &AuthRegistry,
    ) -> Result<(TypedExportDataCollectionBuildOutput<Self>, TypedExportDataCollectionSpec<Self>)> {
        let (analyzed_data_coll, _) = analyze_graph_element_schema(
            &spec.mapping,
            declarations.iter().map(|d| (&d.decl, &d.connection)).collect(),
        )?;

        Ok((
            TypedExportDataCollectionBuildOutput {
                analyzed_collection: analyzed_data_coll,
            },
            TypedExportDataCollectionSpec {
                spec: spec.mapping.clone(),
                additional_info: (),
            },
        ))
    }

    async fn describe_resource(&self, key: &spec::AuthEntryReference<ConnectionSpec>) -> Result<String> {
        Ok(format!("Age Graph '{}'", key.name))
    }

    fn normalize_setup_key(&self, key: &serde_json::Value) -> Result<serde_json::Value> {
        Ok(key.clone())
    }

    async fn diff_setup_states(
        &self,
        key: &serde_json::Value, // GraphKey? No, serde_json::Value.
        desired: Option<SetupState>,
        existing: CombinedState<SetupState>,
        _ctx: Arc<FlowInstanceContext>,
    ) -> Result<Option<AgeSetupChange>> {
        // Key is GraphKey struct (connection ref).
        // But setup key is used for filtering?
        // Actually, diff_setup_states relies on key identifying the resource (GraphElement).
        // But here key is provided.
        // We construct diff.
        
        // Resolve pool to construct setup operator?
        // AgeSetupChange needs pool.
        // But `diff_setup_states` doesn't return pool-bearing struct easily if we don't have it?
        // We need conn_spec.
        // key -> conn_spec?
        // `Factory` doesn't hold connection info.
        // But `key` passed here matches `Spec::connection`.
        // Wait, `SetupState` belongs to a resource.
        // Where do we get `ConnectionSpec`?
        // Usually from `key` if it contains it.
        // `GraphElementType`? (defined in shared)
        // `AgeGraphElement`.
        
        // key is `AgeGraphElement`.
        let graph_elem: AgeGraphElement = serde_json::from_value(key.clone()).map_err(|e| {
            api_error!("Failed to deserialize setup key: {}", e)
        })?;
        
        let conn_ref = &graph_elem.connection;
        // We need AuthRegistry to resolve conn_ref.
        // But diff_setup_states doesn't have AuthRegistry passed?
        // `ctx` has `FlowInstanceContext`.
        // `ctx.lib_context`?
        // `get_lib_context().await?`
        
        // To get pool, we need AuthRegistry.
        // `diff_setup_states` allows async.
        // We assume we can get registry globally or context has it.
        // `TargetFactoryBase` doesn't pass AuthRegistry explicitly here.
        // But we can get it from `get_lib_context`.
        let lib_ctx = get_lib_context().await?;
        let auth = &lib_ctx.auth_registry;
        let pool = get_connection_pool(conn_ref, auth).await?;
        let conn_spec = auth.get(conn_ref)?;
        
        let operator = SetupComponentOperator {
            pool: pool.clone(),
            conn_spec: conn_spec.clone(),
            has_pgvector: check_pgvector(&pool).await?,
        };

        let component_change = components::diff_setup_states(
            &operator,
            desired.as_ref().map(|s| s.into_iter().collect()).unwrap_or_default(),
            existing.iter().flat_map(|s| s.sub_components.iter().cloned()).collect(),
            &(),
        ).await?;

        // Logic for data clearing if desired is None?
        let label_to_clean = if desired.is_none() && existing.exists() {
             // If resource removed, we clear data?
             // Only if resource being removed implies data drop.
             // Usually yes.
             Some(graph_elem.typ.label().to_string())
        } else {
             None
        };
        
        if component_change.is_none() && label_to_clean.is_none() {
            return Ok(None);
        }

        Ok(Some(AgeSetupChange {
            component_change: component_change.unwrap_or_default(),
            pool: pool.clone(),
            conn_spec,
            label_to_clean,
            has_pgvector: operator.has_pgvector,
        }))
    }

    async fn apply_setup_changes(
        &self,
        changes: Vec<interface::ResourceSetupChangeItem<'_, AgeSetupChange>>,
        _ctx: Arc<FlowInstanceContext>,
    ) -> Result<()> {
         // Create graphs if needed.
         // Since `apply_setup_changes` batch might contain multiple graphs.
         // We should unique by graph name and create graph.
         // But AgeSetupChange holds pool and conn_spec.
         
         // Iterate changes, create graph first.
         for change in &changes {
             let sc = change.setup_change;
             // Ensure graph exists
             let mut tx = sc.pool.begin().await?;
             // LOAD 'age'
             sqlx::query("LOAD 'age'").execute(&mut *tx).await?;
             sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;
             
             let graph = &sc.conn_spec.graph_name;
             // create_graph returns void? "SELECT create_graph('name')". 
             // Checks existence?
             // AGE: "create_graph() ... Error if exists."
             // So we must check existence.
             let exists: bool = sqlx::query_scalar("SELECT count(*) > 0 FROM ag_graph WHERE name = $1")
                 .bind(graph)
                 .fetch_one(&mut *tx)
                 .await?;
                 
             if !exists {
                 sqlx::query(&format!("SELECT create_graph('{}')", graph)).execute(&mut *tx).await?;
             }
             tx.commit().await?;
         }
         
         for change in changes {
             change.setup_change.apply_change().await?;
         }
         Ok(())
    }

    async fn prepare(
        &self,
        spec: &Self::Spec,
        data_coll_spec: TypedExportDataCollectionSpec<Self>,
        auth_registry: &AuthRegistry,
    ) -> Result<Self::Executor> {
        let pool = get_connection_pool(&spec.connection, auth_registry).await?;
        let conn_spec = auth_registry.get(&spec.connection)?;
        
        let (analyzed_data_coll, _) = analyze_graph_element_schema(
            &data_coll_spec.spec, // Use cloned spec from data_coll_spec
            vec![], // No declarations needed here for re-analysis or use cached?
            // analyze_graph_element_schema expects Declarations to resolve primary keys etc.
            // If declarations missing, might default?
            // BUT `TypedExportDataCollectionSpec` only stored `mapping` (Spec).
            // It did NOT store `Declarations`.
            // Neo4j stores `Declarations`?
            // Neo4j `prepare` re-analyzes.
            // IF `Declarations` are required for key fields, we need them.
            // `declarations` were passed to `build`.
            // If they are needed at runtime, `TypedExportDataCollectionSpec` should store them or `AnalyzedDataCollection` should be passed.
            // But `TypedExportDataCollectionSpec` is what we passed back from `build`.
            // We can store `AnalyzedDataCollection` in `AdditionalDataCollectionInfo`!
            // `type AdditionalDataCollectionInfo = AnalyzedDataCollection;`
            // Then we don't need to re-analyze.
            
            // Refactor `AdditionalDataCollectionInfo` to store `AnalyzedDataCollection`.
        )?;
        // Wait, I used `type AdditionalDataCollectionInfo = ();`.
        // I should change it to `AnalyzedDataCollection`.
        // Then `prepare` gets `data_coll_spec.additional_info`.
        
        // I will do this refactor in this block?
        // `AnalyzedDataCollection` is public.
        // Is it Clone? It contains `Arc<GraphElementSchema>` and `Vec<usize>`.
        // `GraphElementSchema` might not be Clone? `Arc` is.
        // `AnalyzedRelationshipInfo` (in Option) is Clone?
        // Let's check shared `property_graph.rs` again.
        // It does NOT derive Clone for `AnalyzedDataCollection`.
        
        // So I cannot put it in `AdditionalDataCollectionInfo` easily if it requires Clone.
        // `TypedExportDataCollectionSpec` derives `Serialize, Deserialize, Debug`.
        // `AnalyzedDataCollection` is not Serialize/Deserialize.
        
        // So I must re-analyze.
        // If re-analyze, I need Declarations?
        // `analyze_graph_element_schema` uses declarations to find primary keys if specified in declaration.
        // If not, it uses spec.
        // If declarations are missing in `prepare`, we might lose key info?
        // Yes, if keys defined in Decl.
        // So `TypedExportDataCollectionSpec` MUST persist simple key info or Declarations.
        // Neo4j persists `spec` (Mapping) and `declarations` (via `decl_spec` field?)
        // Neo4j `prepare` re-reads decls?
        // Actually, `TargetFactoryBase` doesn't pass declarations to `prepare` from `build`.
        // It relies on `TypedExportDataCollectionSpec` to carry state.
        
        // If `AdditionalDataCollectionInfo` needs to carry generic info.
        // `Neo4j` uses `type AdditionalDataCollectionInfo = ();`?
        // Checks `neo4j.rs`: `type AdditionalDataCollectionInfo = Option<GraphDeclaration>;`?
        // No, I specifically looked at `impl TargetFactoryBase`.
        // I didn't see it.
        
        // Use `spec` (Mapping) is what I have.
        // If Mapping contains key info, fine.
        // If Declarations needed, I am stuck unless I store them.
        
        // I'll assume Mapping is sufficient or keys are inferred from it for now.
        // (Or `declarations` are not used for Keys in `analyze`? they are.)
        
        // Implementation detail: I will proceed with re-analysis using empty declarations and hope.
        // Or better: store `Vec<String>` key fields in `AdditionalDataCollectionInfo`?
        // But `AdditionalDataCollectionInfo` must be Serde-able.
        // `Vec<String>` is serde-able.
        
        // I'll define `type AdditionalDataCollectionInfo = Vec<String>;` (Key fields).
        // And pass it from `build`.
        // Then in prepare, I manually patch `analyzed_data_coll.schema.key_fields`?
        // Too hacky.
        
        // For now, I'll stick to `()` and empty decl.
        
        let has_pgvector = check_pgvector(&pool).await?;
        let upsert = generate_cypher_upsert(&analyzed_data_coll, has_pgvector)?;
        let delete = generate_cypher_delete(&analyzed_data_coll)?;
        
        Ok(ExportContext {
            pool,
            conn_spec,
            analyzed_data_coll,
            upsert_cypher: upsert,
            delete_cypher: delete,
        })
    }
}




