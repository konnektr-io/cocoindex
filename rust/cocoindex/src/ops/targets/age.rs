use crate::prelude::*;

use super::shared::property_graph::*;
use crate::setup::components::{self, State};
use crate::ops::sdk::{SetupStateCompatibility, TypedResourceSetupChangeItem};
use crate::setup::{ResourceSetupChange, SetupChangeType};
use crate::{ops::sdk::*, setup::CombinedState};
use crate::settings::DatabaseConnectionSpec;

use crate::ops::shared::postgres::get_db_pool;
use sqlx::PgPool;
use std::fmt::Write;
use std::collections::HashMap;
use indexmap::IndexSet;

#[derive(Debug, Deserialize)]
pub struct Spec {
    pub connection: spec::AuthEntryReference<DatabaseConnectionSpec>,
    pub graph_name: String,
    pub mapping: GraphElementMapping,
}

#[derive(Debug, Deserialize)]
pub struct Declaration {
    pub connection: spec::AuthEntryReference<DatabaseConnectionSpec>,
    pub graph_name: String,
    #[serde(flatten)]
    pub decl: GraphDeclaration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AgeGraphElement {
    pub connection: spec::AuthEntryReference<DatabaseConnectionSpec>,
    pub graph_name: String,
    pub typ: ElementType,
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

async fn try_load_age(tx: &mut sqlx::Transaction<'_, sqlx::Postgres>) -> Result<()> {
    sqlx::query("SAVEPOINT load_age_savepoint").execute(&mut **tx).await?;
    if let Err(e) = sqlx::query("LOAD 'age'").execute(&mut **tx).await {
        tracing::debug!("Failed to LOAD 'age': {}. Trying fallback path...", e);
        sqlx::query("ROLLBACK TO SAVEPOINT load_age_savepoint").execute(&mut **tx).await?;
        
        sqlx::query("SAVEPOINT load_age_plugins_savepoint").execute(&mut **tx).await?;
        if let Err(e2) = sqlx::query("LOAD '$libdir/plugins/age'").execute(&mut **tx).await {
             tracing::warn!("Failed to LOAD '$libdir/plugins/age': {}. Proceeding assuming it's preloaded.", e2);
             sqlx::query("ROLLBACK TO SAVEPOINT load_age_plugins_savepoint").execute(&mut **tx).await?;
        } else {
             sqlx::query("RELEASE SAVEPOINT load_age_plugins_savepoint").execute(&mut **tx).await?;
        }
    } else {
        sqlx::query("RELEASE SAVEPOINT load_age_savepoint").execute(&mut **tx).await?;
    }
    Ok(())
}

#[derive(Clone)]
struct SetupComponentOperator {
    pool: PgPool,
    graph_name: String,
    has_pgvector: bool,
}

impl SetupComponentOperator {
    async fn get_graph_oid(&self, tx: &mut sqlx::Transaction<'_, sqlx::Postgres>) -> Result<u32> {
        let oid: i32 = sqlx::query_scalar("SELECT graphid::int FROM ag_catalog.ag_graph WHERE name::text = $1::text")
            .bind(&self.graph_name)
            .fetch_one(&mut **tx)
            .await?;
        Ok(oid as u32)
    }

    async fn ensure_label_exists(
        &self, 
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>, 
        label: &str, 
        is_edge: bool
    ) -> Result<()> {
        let graph = &self.graph_name;
        
        let create_q = if is_edge {
             format!("SELECT create_elabel('{}', '{}')", graph, label)
         } else {
             format!("SELECT create_vlabel('{}', '{}')", graph, label)
         };

         sqlx::query("SAVEPOINT create_label_savepoint").execute(&mut **tx).await?;
         
         if let Err(e) = sqlx::query(&create_q).execute(&mut **tx).await {
             let msg = e.to_string();
             if msg.contains("already exists") {
                 sqlx::query("ROLLBACK TO SAVEPOINT create_label_savepoint").execute(&mut **tx).await?;
             } else {
                 sqlx::query("ROLLBACK TO SAVEPOINT create_label_savepoint").execute(&mut **tx).await?;
                 return Err(e.into());
             }
         } else {
             sqlx::query("RELEASE SAVEPOINT create_label_savepoint").execute(&mut **tx).await?;
         }
        Ok(())
    }
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

    fn is_up_to_date(
        &self,
        current: &Self::State,
        desired: &Self::State,
    ) -> bool {
        current == desired
    }

    async fn create(&self, state: &Self::State, _context: &Self::Context) -> Result<()> {
         if matches!(state.index_def, IndexDef::VectorIndex { .. }) && !self.has_pgvector {
            tracing::warn!("Skipping vector index creation as pgvector extension is missing");
            return Ok(())
         }

         let mut tx = self.pool.begin().await?;
         try_load_age(&mut tx).await?;
         sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

         let graph = &self.graph_name;
         let label = state.object_label.label();
         let is_edge = matches!(state.object_label, ElementType::Relationship(_));

         // Ensure label exists before creating index on it
         self.ensure_label_exists(&mut tx, label, is_edge).await?;

         let index_name = state.key().name;
         let q_graph = format!("\"{}\"", graph);
         let q_label = format!("\"{}\"", label);
         let q_index = format!("\"{}\"", index_name);

         match &state.index_def {
             IndexDef::KeyConstraint { field_names } => {
                 let exprs = field_names
                    .iter()
                    .map(|f| format!("(ag_catalog.agtype_access_operator(properties, '\"{}\"'::ag_catalog.agtype)::text)", f))
                    .collect::<Vec<_>>()
                    .join(", ");
                 let sql = format!("CREATE UNIQUE INDEX {} ON {}.{} ({})", q_index, q_graph, q_label, exprs);
                 tracing::debug!("Executing AGE SQL: {}", sql);
                 sqlx::query(&sql).execute(&mut *tx).await?;
             }
             IndexDef::VectorIndex { field_name, metric, method, vector_size } => {
                 let graph_oid = self.get_graph_oid(&mut tx).await?;
                 
                 let ops = match metric {
                    spec::VectorSimilarityMetric::L2Distance => "vector_l2_ops",
                    spec::VectorSimilarityMetric::CosineSimilarity => "vector_cosine_ops",
                    spec::VectorSimilarityMetric::InnerProduct => "vector_ip_ops",
                 };
                 let method_str = match method {
                     Some(spec::VectorIndexMethod::Hnsw { .. }) | None => "hnsw",
                     Some(spec::VectorIndexMethod::IvfFlat { .. }) => "ivfflat",
                 };
                 
                 // Handle method options
                 let options = match method {
                     Some(spec::VectorIndexMethod::Hnsw { m, ef_construction }) => {
                         let mut opts = Vec::new();
                         if let Some(val) = m { opts.push(format!("m = {}", val)); } 
                         if let Some(val) = ef_construction { opts.push(format!("ef_construction = {}", val)); } 
                         opts
                     },
                     Some(spec::VectorIndexMethod::IvfFlat { lists }) => {
                         let mut opts = Vec::new();
                         if let Some(val) = lists { opts.push(format!("lists = {}", val)); } 
                         opts
                     },
                     None => Vec::new(),
                 };
                 let with_clause = if options.is_empty() { String::new() } else { format!(" WITH ({})", options.join(", ")) };

                 // Complex expression for AGE vector index
                 let expr_inner = format!(
                     r#"agtype_access_operator(VARIADIC ARRAY[_agtype_build_vertex(id, _label_name({}::oid, id), properties), '"{}"'::agtype])::text"#,
                     graph_oid, field_name
                 );
                 let expr = format!("(({})::vector({}))", expr_inner, vector_size);

                 let sql = format!(
                     "CREATE INDEX {} ON {}.{} USING {} ({}) {}{}",
                     q_index, q_graph, q_label, method_str, expr, ops, with_clause
                 );
                 sqlx::query(&sql).execute(&mut *tx).await?;
             }
         }

         tx.commit().await?;
         Ok(())
    }

    async fn delete(&self, key: &Self::Key, _context: &Self::Context) -> Result<()> {
         let mut tx = self.pool.begin().await?;
         let q_graph = format!("\"{}\"", self.graph_name);
         let q_index = format!("\"{}\"", key.name);
         let sql = format!("DROP INDEX IF EXISTS {}.{}", q_graph, q_index);

         sqlx::query(&sql).execute(&mut *tx).await?;
         tx.commit().await?;
         Ok(())
    }
}

pub struct AgeSetupChange {
    component_change: components::SetupChange<SetupComponentOperator>,
    pool: PgPool,
    graph_name: String,
    label_to_clean: Option<DataClearAction>,
}

impl AgeSetupChange {
    async fn apply_change(&self) -> Result<()> {
        if let Some(action) = &self.label_to_clean {
            let mut tx = self.pool.begin().await?;
            try_load_age(&mut tx).await?;
            sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

            let graph = &self.graph_name;
            let cypher = format!("MATCH (n:{}) DETACH DELETE n", action.label);
            let sql = format!("SELECT * FROM cypher('{}'::name, $$ {} $$) as (v agtype)", graph, cypher);
            
            if let Err(e) = sqlx::query(&sql).execute(&mut *tx).await {
                tracing::warn!("Failed to clear data for label {}: {}", action.label, e);
            }

            tx.commit().await?;
        }

        components::apply_component_changes(vec![&self.component_change], &()).await?;

        Ok(())
    }
}

#[derive(Debug, Default)]
struct DataClearAction {
    label: String,
    dependent_node_labels: Vec<String>,
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
        if let Some(action) = &self.label_to_clean {
             let mut desc = format!("Clear data for label '{}'", action.label);
             if !action.dependent_node_labels.is_empty() {
                 write!(&mut desc, "; dependents: {}", action.dependent_node_labels.join(", ")).unwrap();
             }
             changes.push(crate::setup::ChangeDescription::Action(desc));
        }
        changes
    }
}

async fn get_connection_pool(
    conn_ref: &spec::AuthEntryReference<DatabaseConnectionSpec>,
    auth_registry: &AuthRegistry
) -> Result<PgPool> {
    get_db_pool(Some(conn_ref), auth_registry).await
}

async fn check_pgvector(pool: &PgPool) -> Result<bool> {
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        .fetch_one(pool).await.unwrap_or(0);
    Ok(count > 0)
}

pub struct ExportContext {
    pub pool: PgPool,
    pub graph_name: String,
    pub analyzed_data_coll: AnalyzedDataCollection,
    pub upsert_cypher: String,
    pub delete_cypher: String,
}

fn generate_cypher_upsert(coll: &AnalyzedDataCollection, has_pgvector: bool) -> Result<String> {
    use std::fmt::Write;
    let mut q = String::new();
    
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
        ElementType::Relationship(label) => {
            let rel_mapping = coll
                .rel
                .as_ref()
                .ok_or_else(|| api_error!("Missing relationship mapping"))?;

            // Source node
            let src_label = rel_mapping.source.schema.elem_type.label();
            let src_key_match = rel_mapping
                .source
                .schema
                .key_fields
                .iter()
                .map(|f| format!("{}: row.__source.{}", f.name, f.name))
                .collect::<Vec<_>>()
                .join(", ");

            write!(q, "MERGE (start:{} {{ {} }}) ", src_label, src_key_match).unwrap();

            if !rel_mapping.source.schema.value_fields.is_empty() {
                write!(q, "SET ").unwrap();
                let sets = rel_mapping
                    .source
                    .schema
                    .value_fields
                    .iter()
                    .map(|f| {
                        let is_vec = matches!(
                            f.value_type.typ,
                            schema::ValueType::Basic(schema::BasicValueType::Vector(_))
                        );
                        if is_vec && has_pgvector {
                            format!("start.{} = row.__source.{}::vector", f.name, f.name)
                        } else {
                            format!("start.{} = row.__source.{}", f.name, f.name)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(q, "{} ", sets).unwrap();
            }

            // Target node
            let tgt_label = rel_mapping.target.schema.elem_type.label();
            let tgt_key_match = rel_mapping
                .target
                .schema
                .key_fields
                .iter()
                .map(|f| format!("{}: row.__target.{}", f.name, f.name))
                .collect::<Vec<_>>()
                .join(", ");
            write!(q, "MERGE (end:{} {{ {} }}) ", tgt_label, tgt_key_match).unwrap();

            if !rel_mapping.target.schema.value_fields.is_empty() {
                write!(q, "SET ").unwrap();
                let sets = rel_mapping
                    .target
                    .schema
                    .value_fields
                    .iter()
                    .map(|f| {
                        let is_vec = matches!(
                            f.value_type.typ,
                            schema::ValueType::Basic(schema::BasicValueType::Vector(_))
                        );
                        if is_vec && has_pgvector {
                            format!("end.{} = row.__target.{}::vector", f.name, f.name)
                        } else {
                            format!("end.{} = row.__target.{}", f.name, f.name)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(q, "{} ", sets).unwrap();
            }

            // Relationship
            let rel_key_match = coll
                .schema
                .key_fields
                .iter()
                .map(|f| format!("{}: row.{}", f.name, f.name))
                .collect::<Vec<_>>()
                .join(", ");

            write!(
                q,
                "MERGE (start)-[r:{} {{ {} }}]->(end) ",
                label, rel_key_match
            )
            .unwrap();

            if !coll.schema.value_fields.is_empty() {
                write!(q, "SET ").unwrap();
                let sets = coll
                    .schema
                    .value_fields
                    .iter()
                    .map(|f| {
                        let is_vec = matches!(
                            f.value_type.typ,
                            schema::ValueType::Basic(schema::BasicValueType::Vector(_))
                        );
                        if is_vec && has_pgvector {
                            format!("r.{} = row.{}::vector", f.name, f.name)
                        } else {
                            format!("r.{} = row.{}", f.name, f.name)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(q, "{}", sets).unwrap();
            }
        }
    }
    Ok(q)
}

fn generate_cypher_delete(coll: &AnalyzedDataCollection) -> Result<String> {
    use std::fmt::Write;
    let mut q = String::new();
    match &coll.schema.elem_type {
        ElementType::Node(label) => {
            let key_props_match = coll.schema.key_fields.iter() 
                .map(|f| format!("{}: row.{}", f.name, f.name))
                .collect::<Vec<_>>().join(", ");
            write!(q, "MATCH (n:{} {{ {} }}) DETACH DELETE n", label, key_props_match).unwrap();
        }
        ElementType::Relationship(label) => {
            let key_props_match = coll
                .schema
                .key_fields
                .iter()
                .map(|f| format!("{}: row.{}", f.name, f.name))
                .collect::<Vec<_>>()
                .join(", ");
            write!(
                q,
                "MATCH ()-[r:{} {{ {} }}]->() DELETE r",
                label, key_props_match
            )
            .unwrap();
        }
    }
    Ok(q)
}

#[async_trait]
impl TargetFactoryBase for Factory {
    type Spec = Spec;
    type DeclarationSpec = Declaration;
    type SetupState = SetupState;
    type SetupChange = AgeSetupChange;
    type SetupKey = AgeGraphElement;
    type ExportContext = ExportContext;

    fn name(&self) -> &str {
        "Age"
    }

    async fn build(
        self: Arc<Self>,
        data_collections: Vec<TypedExportDataCollectionSpec<Self>>,
        declarations: Vec<Declaration>,
        context: Arc<FlowInstanceContext>,
    ) -> Result<(Vec<TypedExportDataCollectionBuildOutput<Self>>,
        Vec<(AgeGraphElement, SetupState)>)> {
        let (analyzed_data_colls, declared_graph_elements) = analyze_graph_mappings(
            data_collections.iter().map(|d| DataCollectionGraphMappingInput {
                auth_ref: &d.spec.connection,
                mapping: &d.spec.mapping,
                index_options: &d.index_options,
                key_fields_schema: d.key_fields_schema.clone(),
                value_fields_schema: d.value_fields_schema.clone(),
            }),
            declarations.iter().map(|d| (&d.connection, &d.decl)),
        )?;

        let data_coll_output = std::iter::zip(data_collections, analyzed_data_colls)
             .map(|(data_coll, analyzed)| {
                 let setup_key = AgeGraphElement {
                     connection: data_coll.spec.connection.clone(),
                     graph_name: data_coll.spec.graph_name.clone(),
                     typ: analyzed.schema.elem_type.clone(),
                 };
                 let desired_setup_state = SetupState::new(
                     &analyzed.schema,
                     &data_coll.index_options,
                     analyzed
                        .dependent_node_labels()
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect()
                 )?;

                 let conn_spec_ref = data_coll.spec.connection.clone();
                 let graph_name = data_coll.spec.graph_name.clone();
                 let auth_registry = context.auth_registry.clone();
                 
                 let export_context = async move {
                     let pool = get_connection_pool(&conn_spec_ref, &auth_registry).await?;
                     let has_pgvector = check_pgvector(&pool).await?;
                     
                     let upsert = generate_cypher_upsert(&analyzed, has_pgvector)?;
                     let delete = generate_cypher_delete(&analyzed)?;
                     
                     Ok(Arc::new(ExportContext {
                         pool,
                         graph_name,
                         analyzed_data_coll: analyzed,
                         upsert_cypher: upsert,
                         delete_cypher: delete,
                     }))
                 }
                 .boxed();

                 Ok(TypedExportDataCollectionBuildOutput {
                     setup_key,
                     desired_setup_state,
                     export_context,
                 })
             })
             .collect::<Result<Vec<_>>>()?;

        let decl_output = std::iter::zip(declarations, declared_graph_elements)
            .map(|(decl, graph_elem_schema)| {
                let setup_state = SetupState::new(&graph_elem_schema, &decl.decl.index_options, vec![])?;
                let setup_key = AgeGraphElement {
                    connection: decl.connection,
                    graph_name: decl.graph_name,
                    typ: graph_elem_schema.elem_type.clone(),
                };
                Ok((setup_key, setup_state))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok((data_coll_output, decl_output))
    }

    async fn diff_setup_states(
        &self,
        key: AgeGraphElement,
        desired: Option<SetupState>,
        existing: CombinedState<SetupState>,
        context: Arc<FlowInstanceContext>,
    ) -> Result<Self::SetupChange> {
        let pool = get_connection_pool(&key.connection, &context.auth_registry).await?;
        let has_pgvector = check_pgvector(&pool).await?;

        let operator = SetupComponentOperator {
            pool: pool.clone(),
            graph_name: key.graph_name.clone(),
            has_pgvector,
        };

        // Determine data clear actions
        let mut label_to_clean = None;
        if let Some(existing_state) = existing.possible_versions().next() {
            let is_incompatible = if let Some(desired) = &desired {
                desired.check_compatible(existing_state) == SetupStateCompatibility::NotCompatible
            } else {
                true
            };

            if is_incompatible {
                 label_to_clean = Some(DataClearAction {
                     label: key.typ.label().to_string(),
                     dependent_node_labels: existing_state.dependent_node_labels.clone(),
                 });
            }
        }
        
        let component_change = components::SetupChange::create(
            operator,
            desired,
            existing,
        )?;

        Ok(AgeSetupChange {
            component_change,
            pool,
            graph_name: key.graph_name,
            label_to_clean,
        })
    }

    fn check_state_compatibility(
        &self,
        desired: &SetupState,
        existing: &SetupState,
    ) -> Result<SetupStateCompatibility> {
        Ok(desired.check_compatible(existing))
    }

    fn describe_resource(&self, key: &AgeGraphElement) -> Result<String> {
        Ok(format!("Age Graph Element '{}' in Graph '{}' (Connection '{}')", key.typ, key.graph_name, key.connection.key))
    }

    async fn apply_setup_changes(
        &self,
        changes: Vec<TypedResourceSetupChangeItem<'async_trait, Self>>,
        _context: Arc<FlowInstanceContext>,
    ) -> Result<()> {
         // Group by graph to ensure graph exists
         let mut visited_graphs = IndexSet::new(); // Store "graph_name"
         
         for change in &changes {
             let sc = &change.setup_change;
             let key = &sc.graph_name; 
             
             if !visited_graphs.contains(key) {
                 visited_graphs.insert(key.clone());
                 
                 let mut tx = sc.pool.begin().await?;
                 try_load_age(&mut tx).await?;
                 sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;
                 
                 let exists: bool = sqlx::query_scalar("SELECT count(*) > 0 FROM ag_graph WHERE name = $1")
                     .bind(&sc.graph_name)
                     .fetch_one(&mut *tx)
                     .await?;
                     
                 if !exists {
                     sqlx::query(&format!("SELECT create_graph('{}')", sc.graph_name)).execute(&mut *tx).await?;
                 }
                 tx.commit().await?;
             }
         }

         for change in changes {
             change.setup_change.apply_change().await?;
         }
         Ok(())
    }

    async fn apply_mutation(
        &self,
        mutations: Vec<ExportTargetMutationWithContext<'async_trait, ExportContext>>,
    ) -> Result<()> {
        futures::future::try_join_all(mutations.into_iter().map(|m| async move {
             let ctx = m.export_context;
             ctx.upsert(&m.mutation.upserts).await?;
             ctx.delete(&m.mutation.deletes).await?;
             Ok::<(), Error>(())
        })).await?;
        Ok(())
    }
}

// Helper structs for ExportContext
impl ExportContext {
    async fn upsert(&self, upserts: &[ExportTargetUpsertEntry]) -> Result<()> {
        if upserts.is_empty() { return Ok(()); } 
        let mut tx = self.pool.begin().await?;
        try_load_age(&mut tx).await?;
        sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

        // Use PREPARE to handle agtype parameter correctly
        let stmt_name = format!("age_upsert_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
        
        let prepare_sql = format!(
            "PREPARE {} (ag_catalog.agtype) AS SELECT * FROM cypher('{}'::name, $$ UNWIND $batch AS row {} $$::text, $1) as (v ag_catalog.agtype)", 
            stmt_name,
            self.graph_name, 
            self.upsert_cypher
        );
        sqlx::query(&prepare_sql).execute(&mut *tx).await?;

        // Convert to JSON
        let items_json = self.prepare_json(upserts, true)?;
        
        let exec_sql = format!("EXECUTE {} ($1)", stmt_name);
        sqlx::query(&exec_sql).bind(items_json).execute(&mut *tx).await?;
        
        sqlx::query(&format!("DEALLOCATE {}", stmt_name)).execute(&mut *tx).await?;

        tx.commit().await?;
        Ok(())
    }

    async fn delete(&self, deletes: &[ExportTargetDeleteEntry]) -> Result<()> {
        if deletes.is_empty() { return Ok(()); } 
        let mut tx = self.pool.begin().await?;
        try_load_age(&mut tx).await?;
        sqlx::query("SET LOCAL search_path = ag_catalog, \"$user\", public").execute(&mut *tx).await?;

        // Convert to JSON
        let items_json = self.prepare_json_delete(deletes)?;

        let sql = format!(
            "SELECT * FROM cypher('{}'::name, $$ UNWIND $batch AS row {} $$, $1::agtype) as (v agtype)", 
            self.graph_name, 
            self.delete_cypher
        );
        sqlx::query(&sql).bind(items_json).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    fn prepare_json(
        &self,
        items: &[ExportTargetUpsertEntry],
        include_values: bool,
    ) -> Result<String> {
        let mut rows = Vec::with_capacity(items.len());
        for item in items {
            let mut row_map = serde_json::Map::new();
            self.fill_key(&mut row_map, &item.key)?;

            if include_values {
                for (schema_idx, &val_idx) in self
                    .analyzed_data_coll
                    .value_fields_input_idx
                    .iter()
                    .enumerate()
                {
                    let field = &self.analyzed_data_coll.schema.value_fields[schema_idx];
                    let val = &item.value.fields[val_idx];
                    row_map.insert(field.name.clone(), serde_json::to_value(val)?);
                }

                if let Some(rel_mapping) = &self.analyzed_data_coll.rel {
                    let mut src_map = serde_json::Map::new();
                    for (schema_idx, &val_idx) in
                        rel_mapping.source.fields_input_idx.key.iter().enumerate()
                    {
                        let field = &rel_mapping.source.schema.key_fields[schema_idx];
                        let val = &item.value.fields[val_idx];
                        src_map.insert(field.name.clone(), serde_json::to_value(val)?);
                    }
                    for (schema_idx, &val_idx) in
                        rel_mapping.source.fields_input_idx.value.iter().enumerate()
                    {
                        let field = &rel_mapping.source.schema.value_fields[schema_idx];
                        let val = &item.value.fields[val_idx];
                        src_map.insert(field.name.clone(), serde_json::to_value(val)?);
                    }
                    row_map.insert("__source".to_string(), serde_json::Value::Object(src_map));

                    let mut tgt_map = serde_json::Map::new();
                    for (schema_idx, &val_idx) in
                        rel_mapping.target.fields_input_idx.key.iter().enumerate()
                    {
                        let field = &rel_mapping.target.schema.key_fields[schema_idx];
                        let val = &item.value.fields[val_idx];
                        tgt_map.insert(field.name.clone(), serde_json::to_value(val)?);
                    }
                    for (schema_idx, &val_idx) in
                        rel_mapping.target.fields_input_idx.value.iter().enumerate()
                    {
                        let field = &rel_mapping.target.schema.value_fields[schema_idx];
                        let val = &item.value.fields[val_idx];
                        tgt_map.insert(field.name.clone(), serde_json::to_value(val)?);
                    }
                    row_map.insert("__target".to_string(), serde_json::Value::Object(tgt_map));
                }
            }
            rows.push(serde_json::Value::Object(row_map));
        }
        Ok(serde_json::to_string(&serde_json::json!({ "batch": rows }))?)
    }

    fn prepare_json_delete(&self, items: &[ExportTargetDeleteEntry]) -> Result<String> {
        let mut rows = Vec::with_capacity(items.len());
        for item in items {
            let mut row_map = serde_json::Map::new();
            self.fill_key(&mut row_map, &item.key)?;
            rows.push(serde_json::Value::Object(row_map));
        }
        Ok(serde_json::to_string(&serde_json::json!({ "batch": rows }))?)
    }

    fn fill_key(&self, row_map: &mut serde_json::Map<String, serde_json::Value>, key: &KeyValue) -> Result<()> {
         for (i, part) in key.iter().enumerate() {
            if i < self.analyzed_data_coll.schema.key_fields.len() {
                let field = &self.analyzed_data_coll.schema.key_fields[i];
                row_map.insert(field.name.clone(), serde_json::to_value(part)?);
            }
         }
        Ok(())
    }
}
