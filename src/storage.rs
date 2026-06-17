/// Rust guideline compliant 2026-06-17
use crate::bm25::{Bm25Config, Bm25Index};
use crate::okf::OkfDocument;
use crate::query::SearchResult;
use anyhow::{anyhow, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// On-disk index format version.
pub const INDEX_FORMAT_VERSION: u32 = 4;
/// Magic bytes identifying a segment file.
pub const SEGMENT_MAGIC: &[u8; 8] = b"OKFSEG04";
/// Magic bytes identifying a journal file.
pub const JOURNAL_MAGIC: &[u8; 8] = b"OKFJRN01";

/// Manifest describing the current index state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub generation: u64,
    pub embedding_model: String,
    pub embedding_dimension: usize,
    pub bm25: Bm25Config,
    pub segments: Vec<SegmentEntry>,
    pub tombstones: Vec<String>,
}

/// Manifest entry for a single immutable segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentEntry {
    pub segment_id: String,
    pub path: String,
    pub doc_count: usize,
    pub created_at: u64,
}

/// Recovery journal entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JournalEntry {
    BeginCommit { segment_id: String },
    SegmentWritten { segment_id: String, path: String },
    ManifestWritten { generation: u64 },
}

/// Recovery journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Journal {
    pub format_version: u32,
    pub entries: Vec<JournalEntry>,
}

impl Default for Journal {
    fn default() -> Self {
        Self {
            format_version: INDEX_FORMAT_VERSION,
            entries: Vec::new(),
        }
    }
}

/// Mmap-backed storage for a segment file.
#[derive(Clone)]
pub struct SegmentView {
    mmap: Arc<Mmap>,
    header: SegmentHeader,
}

/// Borrowed view over a single document inside a segment.
pub struct DocumentView<'a> {
    strings: &'a [u8],
    entry: DocEntry,
}

/// Per-segment summary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMetadata {
    pub doc_count: usize,
    pub avg_doc_len: f32,
    pub embedding_dimension: usize,
    pub created_at: u64,
}

/// Read-only access to a loaded segment.
pub trait SegmentReader {
    fn metadata(&self) -> &SegmentMetadata;
    fn documents(&self) -> Vec<OkfDocument>;
    fn bm25(&self) -> Bm25Index;
    fn embeddings(&self) -> Vec<Vec<f32>>;
}

/// Materialized segment payload used by indexing.
#[derive(Debug, Clone)]
pub struct SegmentFile {
    pub segment_id: String,
    pub metadata: SegmentMetadata,
    pub documents: Vec<OkfDocument>,
    pub bm25: Bm25Index,
    pub embeddings: Vec<Vec<f32>>,
}

impl SegmentReader for SegmentFile {
    fn metadata(&self) -> &SegmentMetadata {
        &self.metadata
    }

    fn documents(&self) -> Vec<OkfDocument> {
        self.documents.clone()
    }

    fn bm25(&self) -> Bm25Index {
        self.bm25.clone()
    }

    fn embeddings(&self) -> Vec<Vec<f32>> {
        self.embeddings.clone()
    }
}

/// File-backed storage for manifests, journals, and segments.
pub struct IndexStorage {
    root: PathBuf,
}

impl IndexStorage {
    /// Opens or creates the index root directory.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("segments"))?;
        fs::create_dir_all(root.join("journal"))?;
        Ok(Self { root })
    }

    /// Returns the manifest file path.
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }

    /// Returns the journal file path.
    pub fn journal_path(&self) -> PathBuf {
        self.root.join("journal").join("journal.bin")
    }

    /// Loads the manifest from disk.
    pub fn load_manifest(&self) -> Result<Manifest> {
        let path = self.manifest_path();
        if !path.exists() {
            return Ok(Manifest {
                format_version: INDEX_FORMAT_VERSION,
                generation: 0,
                embedding_model: "unknown".to_string(),
                embedding_dimension: 0,
                bm25: Bm25Config::default(),
                segments: vec![],
                tombstones: vec![],
            });
        }
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    /// Saves the manifest atomically.
    pub fn save_manifest(&self, manifest: &Manifest) -> Result<()> {
        let tmp = self.root.join("manifest.json.tmp");
        fs::write(&tmp, serde_json::to_string_pretty(manifest)?)?;
        fs::rename(tmp, self.manifest_path())?;
        self.append_journal(JournalEntry::ManifestWritten {
            generation: manifest.generation,
        })?;
        Ok(())
    }

    /// Reads the recovery journal.
    pub fn load_journal(&self) -> Result<Journal> {
        let path = self.journal_path();
        if !path.exists() {
            return Ok(Journal::default());
        }
        let bytes = fs::read(path)?;
        let (journal, _): (Journal, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
        Ok(journal)
    }

    /// Appends a journal entry.
    pub fn append_journal(&self, entry: JournalEntry) -> Result<()> {
        let mut journal = self.load_journal()?;
        journal.entries.push(entry);
        let tmp = self.root.join("journal").join("journal.bin.tmp");
        let bytes = bincode::serde::encode_to_vec(journal, bincode::config::standard())?;
        fs::write(&tmp, bytes)?;
        fs::rename(tmp, self.journal_path())?;
        Ok(())
    }

    /// Clears the recovery journal.
    pub fn clear_journal(&self) -> Result<()> {
        let path = self.journal_path();
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Writes a segment to disk.
    pub fn write_segment(&self, segment: &SegmentFile) -> Result<SegmentEntry> {
        let staging_dir = self
            .root
            .join("segments")
            .join(format!("{}.staging", segment.segment_id));
        let final_dir = self.root.join("segments").join(&segment.segment_id);
        if staging_dir.exists() {
            fs::remove_dir_all(&staging_dir)?;
        }
        fs::create_dir_all(&staging_dir)?;
        self.append_journal(JournalEntry::BeginCommit {
            segment_id: segment.segment_id.clone(),
        })?;
        let tmp_file = staging_dir.join("segment.bin.tmp");
        write_segment_bin(&tmp_file, segment)?;
        fs::rename(&tmp_file, staging_dir.join("segment.bin"))?;
        fs::rename(&staging_dir, &final_dir)?;
        self.append_journal(JournalEntry::SegmentWritten {
            segment_id: segment.segment_id.clone(),
            path: final_dir.to_string_lossy().to_string(),
        })?;
        Ok(SegmentEntry {
            segment_id: segment.segment_id.clone(),
            path: final_dir.to_string_lossy().to_string(),
            doc_count: segment.metadata.doc_count,
            created_at: segment.metadata.created_at,
        })
    }

    /// Loads a segment from disk as a zero-copy mmap view.
    pub fn read_segment(&self, entry: &SegmentEntry) -> Result<SegmentView> {
        SegmentView::open(Path::new(&entry.path).join("segment.bin"))
    }

    /// Opens a segment and materializes it into owned data.
    pub fn read_segment_owned(&self, entry: &SegmentEntry) -> Result<SegmentFile> {
        self.read_segment(entry)?.materialize()
    }

    /// Returns the index root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Recovers after an interrupted write.
    pub fn recover(&self, manifest: &mut Manifest) -> Result<()> {
        let journal = self.load_journal()?;
        let mut repaired = false;
        for entry in journal.entries {
            match entry {
                JournalEntry::BeginCommit { segment_id } => {
                    let staging_dir = self
                        .root
                        .join("segments")
                        .join(format!("{}.staging", segment_id));
                    if staging_dir.exists() {
                        fs::remove_dir_all(staging_dir)?;
                        repaired = true;
                    }
                }
                JournalEntry::SegmentWritten { segment_id, path } => {
                    if !manifest
                        .segments
                        .iter()
                        .any(|seg| seg.segment_id == segment_id)
                    {
                        let recovered = SegmentView::open(Path::new(&path).join("segment.bin"))?
                            .materialize()?;
                        manifest.segments.push(SegmentEntry {
                            segment_id,
                            path,
                            doc_count: recovered.metadata.doc_count,
                            created_at: recovered.metadata.created_at,
                        });
                        repaired = true;
                    }
                }
                JournalEntry::ManifestWritten { generation } => {
                    if manifest.generation < generation {
                        manifest.generation = generation;
                        repaired = true;
                    }
                }
            }
        }
        if repaired {
            self.save_manifest(manifest)?;
        }
        self.clear_journal()?;
        Ok(())
    }

    /// Compacts live segments into a new segment.
    pub fn compact(&self, segments: &[SegmentView], manifest: &Manifest) -> Result<SegmentEntry> {
        let live_docs: Vec<OkfDocument> = segments
            .iter()
            .flat_map(|segment| {
                segment
                    .document_views()
                    .into_iter()
                    .map(|doc| doc.to_owned())
            })
            .filter(|doc| !manifest.tombstones.iter().any(|dead| dead == &doc.doc_id))
            .collect();
        if live_docs.is_empty() {
            return Err(anyhow!("nothing to compact"));
        }
        let segment_id = format!("compact_{:016x}", manifest.generation + 1);
        let embeddings = segments
            .iter()
            .flat_map(|segment| segment.embeddings().into_iter())
            .take(live_docs.len())
            .collect::<Vec<_>>();
        let bm25_pairs: Vec<(String, String)> = live_docs
            .iter()
            .map(|doc| (doc.doc_id.clone(), doc.searchable_text.clone()))
            .collect();
        let bm25 = Bm25Index::build(&bm25_pairs, manifest.bm25.clone());
        let segment = SegmentFile {
            segment_id: segment_id.clone(),
            metadata: SegmentMetadata {
                doc_count: live_docs.len(),
                avg_doc_len: bm25.avg_doc_len,
                embedding_dimension: manifest.embedding_dimension,
                created_at: manifest.generation + 1,
            },
            documents: live_docs,
            bm25,
            embeddings,
        };
        self.write_segment(&segment)
    }
}

impl SegmentView {
    /// Opens a segment file as an mmap-backed view.
    pub fn open(path: PathBuf) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = map_file(&file)?;
        let header = parse_segment_header(&mmap)?;
        Ok(Self {
            mmap: Arc::new(mmap),
            header,
        })
    }

    /// Materializes the view into an owned segment payload.
    pub fn materialize(&self) -> Result<SegmentFile> {
        let documents = self
            .document_views()
            .into_iter()
            .map(|doc| doc.to_owned())
            .collect::<Vec<_>>();
        let bm25 = self.bm25();
        let embeddings = self.embeddings();
        Ok(SegmentFile {
            segment_id: self.header.segment_id.clone(),
            metadata: SegmentMetadata {
                doc_count: documents.len(),
                avg_doc_len: bm25.avg_doc_len,
                embedding_dimension: self.header.vector_dim,
                created_at: self.header.metadata.created_at,
            },
            documents,
            bm25,
            embeddings,
        })
    }

    /// Returns the borrowed document views.
    pub fn document_views(&self) -> Vec<DocumentView<'_>> {
        let docs_bytes = self.slice(self.header.docs_offset, self.header.docs_len);
        let mut cursor = 0usize;
        let mut views = Vec::with_capacity(self.header.doc_count);
        for _ in 0..self.header.doc_count {
            let entry = DocEntry {
                logical_key_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                logical_key_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                bundle_path_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                bundle_path_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                concept_path_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                concept_path_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                file_path_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                file_path_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                doc_id_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                doc_id_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                type_name_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                type_name_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                title_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                title_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                description_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                description_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                resource_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                resource_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                timestamp_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                timestamp_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                tags_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                tags_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                body_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                body_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                searchable_off: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
                searchable_len: read_u32(docs_bytes, &mut cursor).unwrap_or(0),
            };
            views.push(DocumentView {
                strings: self.strings(),
                entry,
            });
        }
        views
    }

    /// Returns the BM25 state reconstructed from the mmap view.
    pub fn bm25(&self) -> Bm25Index {
        let docs = self.document_views();
        parse_bm25(
            self.strings(),
            self.terms_bytes(),
            self.postings_bytes(),
            &docs,
            self.header.doc_count,
        )
        .unwrap_or_else(|_| Bm25Index {
            config: Bm25Config::default(),
            doc_len: HashMap::new(),
            avg_doc_len: 0.0,
            doc_count: self.header.doc_count,
            term_doc_freq: HashMap::new(),
            postings: HashMap::new(),
        })
    }

    /// Returns the dense vectors reconstructed from the mmap view.
    pub fn embeddings(&self) -> Vec<Vec<f32>> {
        parse_embeddings(
            self.vectors_bytes(),
            self.header.doc_count,
            self.header.vector_dim,
        )
        .unwrap_or_default()
    }

    fn strings(&self) -> &[u8] {
        self.slice(self.header.strings_offset, self.header.strings_len)
    }

    fn terms_bytes(&self) -> &[u8] {
        self.slice(self.header.terms_offset, self.header.terms_len)
    }

    fn postings_bytes(&self) -> &[u8] {
        self.slice(self.header.postings_offset, self.header.postings_len)
    }

    fn vectors_bytes(&self) -> &[u8] {
        self.slice(self.header.vectors_offset, self.header.vectors_len)
    }

    fn slice(&self, offset: u64, len: u64) -> &[u8] {
        let start = offset as usize;
        let end = (offset + len) as usize;
        &self.mmap[start..end]
    }
}

impl<'a> DocumentView<'a> {
    fn get(&self, off: u32, len: u32) -> Option<&'a str> {
        read_string(self.strings, off, len)
    }

    /// Returns the logical key.
    pub fn logical_key(&self) -> Option<&'a str> {
        self.get(self.entry.logical_key_off, self.entry.logical_key_len)
    }

    pub fn doc_id(&self) -> Option<&'a str> {
        self.get(self.entry.doc_id_off, self.entry.doc_id_len)
    }

    pub fn title(&self) -> Option<&'a str> {
        self.get(self.entry.title_off, self.entry.title_len)
    }

    pub fn searchable_text(&self) -> Option<&'a str> {
        self.get(self.entry.searchable_off, self.entry.searchable_len)
    }

    pub fn concept_path(&self) -> Option<&'a str> {
        self.get(self.entry.concept_path_off, self.entry.concept_path_len)
    }

    pub fn body(&self) -> Option<&'a str> {
        self.get(self.entry.body_off, self.entry.body_len)
    }

    pub fn bundle_path(&self) -> Option<&'a str> {
        self.get(self.entry.bundle_path_off, self.entry.bundle_path_len)
    }

    pub fn file_path(&self) -> Option<&'a str> {
        self.get(self.entry.file_path_off, self.entry.file_path_len)
    }

    pub fn type_name(&self) -> Option<&'a str> {
        self.get(self.entry.type_name_off, self.entry.type_name_len)
    }

    pub fn description(&self) -> Option<&'a str> {
        self.get(self.entry.description_off, self.entry.description_len)
    }

    pub fn resource(&self) -> Option<&'a str> {
        self.get(self.entry.resource_off, self.entry.resource_len)
    }

    pub fn timestamp(&self) -> Option<&'a str> {
        self.get(self.entry.timestamp_off, self.entry.timestamp_len)
    }

    pub fn tags(&self) -> Vec<String> {
        self.get(self.entry.tags_off, self.entry.tags_len)
            .map(|s| s.split('\n').map(str::to_string).collect())
            .unwrap_or_default()
    }

    pub fn to_owned(&self) -> OkfDocument {
        OkfDocument {
            logical_key: self.logical_key().unwrap_or_default().to_string(),
            bundle_path: self.bundle_path().unwrap_or_default().to_string(),
            concept_path: self.concept_path().unwrap_or_default().to_string(),
            file_path: self.file_path().unwrap_or_default().to_string(),
            doc_id: self.doc_id().unwrap_or_default().to_string(),
            type_name: self.type_name().unwrap_or_default().to_string(),
            title: self.title().map(str::to_string),
            description: self.description().map(str::to_string),
            resource: self.resource().map(str::to_string),
            tags: self.tags(),
            timestamp: self.timestamp().map(str::to_string),
            body: self.body().unwrap_or_default().to_string(),
            searchable_text: self.searchable_text().unwrap_or_default().to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct SegmentHeader {
    segment_id: String,
    metadata: SegmentMetadata,
    doc_count: usize,
    vector_dim: usize,
    docs_offset: u64,
    docs_len: u64,
    strings_offset: u64,
    strings_len: u64,
    terms_offset: u64,
    terms_len: u64,
    postings_offset: u64,
    postings_len: u64,
    vectors_offset: u64,
    vectors_len: u64,
}

fn parse_segment_header(mmap: &Mmap) -> Result<SegmentHeader> {
    if mmap.get(0..8) != Some(SEGMENT_MAGIC.as_slice()) {
        return Err(anyhow!("invalid segment magic"));
    }
    let mut cursor = 8usize;
    let version = read_u32(mmap, &mut cursor)?;
    if version != INDEX_FORMAT_VERSION {
        return Err(anyhow!("unsupported segment version"));
    }
    let doc_count = read_u32(mmap, &mut cursor)? as usize;
    let _term_count = read_u32(mmap, &mut cursor)? as usize;
    let _posting_count = read_u32(mmap, &mut cursor)? as usize;
    let vector_dim = read_u32(mmap, &mut cursor)? as usize;
    let docs_offset = read_u64(mmap, &mut cursor)?;
    let docs_len = read_u64(mmap, &mut cursor)?;
    let strings_offset = read_u64(mmap, &mut cursor)?;
    let strings_len = read_u64(mmap, &mut cursor)?;
    let terms_offset = read_u64(mmap, &mut cursor)?;
    let terms_len = read_u64(mmap, &mut cursor)?;
    let postings_offset = read_u64(mmap, &mut cursor)?;
    let postings_len = read_u64(mmap, &mut cursor)?;
    let vectors_offset = read_u64(mmap, &mut cursor)?;
    let vectors_len = read_u64(mmap, &mut cursor)?;
    let segment_id = format!("mmap-{:x}", docs_offset);
    Ok(SegmentHeader {
        segment_id,
        metadata: SegmentMetadata {
            doc_count,
            avg_doc_len: 0.0,
            embedding_dimension: vector_dim,
            created_at: 0,
        },
        doc_count,
        vector_dim,
        docs_offset,
        docs_len,
        strings_offset,
        strings_len,
        terms_offset,
        terms_len,
        postings_offset,
        postings_len,
        vectors_offset,
        vectors_len,
    })
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = *cursor + 4;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("unexpected eof"))?;
    *cursor = end;
    let array: [u8; 4] = slice
        .try_into()
        .map_err(|_| anyhow!("invalid u32 encoding"))?;
    Ok(u32::from_le_bytes(array))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = *cursor + 8;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("unexpected eof"))?;
    *cursor = end;
    let array: [u8; 8] = slice
        .try_into()
        .map_err(|_| anyhow!("invalid u64 encoding"))?;
    Ok(u64::from_le_bytes(array))
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64_to(file: &mut File, v: u64) -> Result<()> {
    file.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn push_string(pool: &mut Vec<u8>, s: Option<&str>) -> (u32, u32) {
    match s {
        Some(text) => {
            let off = pool.len() as u32;
            pool.extend_from_slice(text.as_bytes());
            pool.push(0);
            (off, text.len() as u32)
        }
        None => (u32::MAX, 0),
    }
}

fn read_string(pool: &[u8], off: u32, len: u32) -> Option<&str> {
    if off == u32::MAX {
        return None;
    }
    std::str::from_utf8(pool.get(off as usize..off as usize + len as usize)?).ok()
}

fn write_segment_bin(path: &Path, segment: &SegmentFile) -> Result<()> {
    let mut strings = Vec::new();
    let mut docs = Vec::new();
    let mut doc_offsets = HashMap::new();

    for (i, doc) in segment.documents.iter().enumerate() {
        doc_offsets.insert(doc.doc_id.clone(), i as u32);
        let tags = if doc.tags.is_empty() {
            None
        } else {
            Some(doc.tags.join("\n"))
        };
        let searchable = Some(doc.searchable_text.as_str());
        let entries = [
            push_string(&mut strings, Some(&doc.logical_key)),
            push_string(&mut strings, Some(&doc.bundle_path)),
            push_string(&mut strings, Some(&doc.concept_path)),
            push_string(&mut strings, Some(&doc.file_path)),
            push_string(&mut strings, Some(&doc.doc_id)),
            push_string(&mut strings, Some(&doc.type_name)),
            push_string(&mut strings, doc.title.as_deref()),
            push_string(&mut strings, doc.description.as_deref()),
            push_string(&mut strings, doc.resource.as_deref()),
            push_string(&mut strings, doc.timestamp.as_deref()),
            push_string(&mut strings, tags.as_deref()),
            push_string(&mut strings, Some(&doc.body)),
            push_string(&mut strings, searchable),
        ];
        let [logical, bundle, concept, file, doc_id, type_name, title, description, resource, timestamp, tags, body, searchable] =
            entries;
        docs.push(DocEntry {
            logical_key_off: logical.0,
            logical_key_len: logical.1,
            bundle_path_off: bundle.0,
            bundle_path_len: bundle.1,
            concept_path_off: concept.0,
            concept_path_len: concept.1,
            file_path_off: file.0,
            file_path_len: file.1,
            doc_id_off: doc_id.0,
            doc_id_len: doc_id.1,
            type_name_off: type_name.0,
            type_name_len: type_name.1,
            title_off: title.0,
            title_len: title.1,
            description_off: description.0,
            description_len: description.1,
            resource_off: resource.0,
            resource_len: resource.1,
            timestamp_off: timestamp.0,
            timestamp_len: timestamp.1,
            tags_off: tags.0,
            tags_len: tags.1,
            body_off: body.0,
            body_len: body.1,
            searchable_off: searchable.0,
            searchable_len: searchable.1,
        });
    }

    let mut term_map: Vec<(String, Vec<PostingEntry>)> = Vec::new();
    for (term, postings) in &segment.bm25.postings {
        let mut list = Vec::new();
        for (doc_id, tf) in postings {
            let idx = *doc_offsets
                .get(doc_id)
                .ok_or_else(|| anyhow!("missing doc id"))?;
            list.push(PostingEntry {
                doc_index: idx,
                tf: *tf as u32,
            });
        }
        list.sort_by_key(|p| p.doc_index);
        term_map.push((term.clone(), list));
    }
    term_map.sort_by(|a, b| a.0.cmp(&b.0));

    let mut terms = Vec::new();
    let mut postings = Vec::new();
    for (term, list) in term_map {
        let (term_off, term_len) = push_string(&mut strings, Some(&term));
        let postings_off = postings.len() as u32;
        for p in list {
            write_u32(&mut postings, p.doc_index);
            write_u32(&mut postings, p.tf);
        }
        let postings_len = (postings.len() as u32) - postings_off;
        terms.push(TermEntry {
            term_off,
            term_len,
            postings_off,
            postings_len,
        });
    }

    let mut vectors = Vec::new();
    for vec in &segment.embeddings {
        for f in vec {
            vectors.extend_from_slice(&f.to_le_bytes());
        }
    }

    let header_size = 8 + 4 + 4 + 4 + 4 + 4 + 10 * 8;
    let docs_offset = header_size as u64;
    let docs_len = (docs.len() * std::mem::size_of::<DocEntry>()) as u64;
    let strings_offset = docs_offset + docs_len;
    let strings_len = strings.len() as u64;
    let terms_offset = strings_offset + strings_len;
    let terms_len = (terms.len() * std::mem::size_of::<TermEntry>()) as u64;
    let postings_offset = terms_offset + terms_len;
    let postings_len = postings.len() as u64;
    let vectors_offset = postings_offset + postings_len;
    let vectors_len = vectors.len() as u64;

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    file.write_all(SEGMENT_MAGIC)?;
    file.write_all(&INDEX_FORMAT_VERSION.to_le_bytes())?;
    file.write_all(&(segment.documents.len() as u32).to_le_bytes())?;
    file.write_all(&(terms.len() as u32).to_le_bytes())?;
    file.write_all(&((postings.len() / 8) as u32).to_le_bytes())?;
    file.write_all(&(segment.metadata.embedding_dimension as u32).to_le_bytes())?;
    for v in [
        docs_offset,
        docs_len,
        strings_offset,
        strings_len,
        terms_offset,
        terms_len,
        postings_offset,
        postings_len,
        vectors_offset,
        vectors_len,
    ] {
        write_u64_to(&mut file, v)?;
    }
    for doc in &docs {
        write_doc_entry(&mut file, doc)?;
    }
    file.write_all(&strings)?;
    for term in &terms {
        write_term_entry(&mut file, term)?;
    }
    file.write_all(&postings)?;
    file.write_all(&vectors)?;
    file.sync_all()?;
    Ok(())
}

fn write_doc_entry(file: &mut File, doc: &DocEntry) -> Result<()> {
    for v in [
        doc.logical_key_off,
        doc.logical_key_len,
        doc.bundle_path_off,
        doc.bundle_path_len,
        doc.concept_path_off,
        doc.concept_path_len,
        doc.file_path_off,
        doc.file_path_len,
        doc.doc_id_off,
        doc.doc_id_len,
        doc.type_name_off,
        doc.type_name_len,
        doc.title_off,
        doc.title_len,
        doc.description_off,
        doc.description_len,
        doc.resource_off,
        doc.resource_len,
        doc.timestamp_off,
        doc.timestamp_len,
        doc.tags_off,
        doc.tags_len,
        doc.body_off,
        doc.body_len,
        doc.searchable_off,
        doc.searchable_len,
    ] {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_term_entry(file: &mut File, term: &TermEntry) -> Result<()> {
    for v in [
        term.term_off,
        term.term_len,
        term.postings_off,
        term.postings_len,
    ] {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn map_file(file: &File) -> Result<Mmap> {
    // SAFETY: The mapping is read-only and used for immutable access only.
    let mmap = unsafe { Mmap::map(file)? };
    Ok(mmap)
}

#[derive(Clone, Copy, Debug)]
struct DocEntry {
    logical_key_off: u32,
    logical_key_len: u32,
    bundle_path_off: u32,
    bundle_path_len: u32,
    concept_path_off: u32,
    concept_path_len: u32,
    file_path_off: u32,
    file_path_len: u32,
    doc_id_off: u32,
    doc_id_len: u32,
    type_name_off: u32,
    type_name_len: u32,
    title_off: u32,
    title_len: u32,
    description_off: u32,
    description_len: u32,
    resource_off: u32,
    resource_len: u32,
    timestamp_off: u32,
    timestamp_len: u32,
    tags_off: u32,
    tags_len: u32,
    body_off: u32,
    body_len: u32,
    searchable_off: u32,
    searchable_len: u32,
}

#[derive(Clone, Copy, Debug)]
struct TermEntry {
    term_off: u32,
    term_len: u32,
    postings_off: u32,
    postings_len: u32,
}

#[derive(Clone, Copy, Debug)]
struct PostingEntry {
    doc_index: u32,
    tf: u32,
}

fn parse_bm25(
    strings: &[u8],
    terms_bytes: &[u8],
    postings_bytes: &[u8],
    docs: &[DocumentView<'_>],
    doc_count: usize,
) -> Result<Bm25Index> {
    let mut bm25 = Bm25Index {
        config: Bm25Config::default(),
        doc_len: HashMap::new(),
        avg_doc_len: 0.0,
        doc_count,
        term_doc_freq: HashMap::new(),
        postings: HashMap::new(),
    };
    let mut total_len = 0usize;
    let mut cursor = 0usize;
    let term_count = terms_bytes.len() / std::mem::size_of::<TermEntry>();
    for _ in 0..term_count {
        let entry = TermEntry {
            term_off: read_u32(terms_bytes, &mut cursor)?,
            term_len: read_u32(terms_bytes, &mut cursor)?,
            postings_off: read_u32(terms_bytes, &mut cursor)?,
            postings_len: read_u32(terms_bytes, &mut cursor)?,
        };
        let term = read_string(strings, entry.term_off, entry.term_len)
            .unwrap_or_default()
            .to_string();
        let start = entry.postings_off as usize;
        let end = start + entry.postings_len as usize;
        let slice = postings_bytes
            .get(start..end)
            .ok_or_else(|| anyhow!("bad postings slice"))?;
        let mut pcur = 0usize;
        let mut postings = HashMap::new();
        while pcur < slice.len() {
            let doc_index = read_u32(slice, &mut pcur)? as usize;
            let tf = read_u32(slice, &mut pcur)? as usize;
            let doc_id = docs
                .get(doc_index)
                .ok_or_else(|| anyhow!("bad doc index"))?
                .doc_id()
                .ok_or_else(|| anyhow!("bad doc id"))?
                .to_string();
            *postings.entry(doc_id).or_insert(0) += tf;
        }
        bm25.term_doc_freq.insert(term.clone(), postings.len());
        bm25.postings.insert(term, postings);
    }

    for doc in docs {
        let searchable_text = doc.searchable_text().unwrap_or_default();
        let len = crate::bm25::tokenize(searchable_text).len();
        bm25.doc_len
            .insert(doc.doc_id().unwrap_or_default().to_string(), len);
        total_len += len;
    }
    bm25.avg_doc_len = if docs.is_empty() {
        0.0
    } else {
        total_len as f32 / docs.len() as f32
    };
    Ok(bm25)
}

fn parse_embeddings(
    vectors_bytes: &[u8],
    doc_count: usize,
    vector_dim: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut embeddings = Vec::with_capacity(doc_count);
    let mut cursor = 0usize;
    for _ in 0..doc_count {
        let mut vec = Vec::with_capacity(vector_dim);
        for _ in 0..vector_dim {
            let bytes = vectors_bytes
                .get(cursor..cursor + 4)
                .ok_or_else(|| anyhow!("bad vector slice"))?;
            let array: [u8; 4] = bytes
                .try_into()
                .map_err(|_| anyhow!("invalid f32 encoding"))?;
            vec.push(f32::from_le_bytes(array));
            cursor += 4;
        }
        embeddings.push(vec);
    }
    Ok(embeddings)
}

/// Extracts a short snippet for display in search results.
pub fn snippet_from_doc(doc: &OkfDocument, query: &str) -> String {
    let lowered = query.to_lowercase();
    let body = &doc.body;
    if let Some(pos) = body.to_lowercase().find(&lowered) {
        let start = pos.saturating_sub(40);
        let end = (pos + lowered.len() + 80).min(body.len());
        return body[start..end].to_string();
    }
    body.chars().take(160).collect()
}

/// Converts indexed documents back into API search results.
pub fn results_from_docs(
    docs: &[OkfDocument],
    bm25: &HashMap<String, f32>,
    vectors: &HashMap<String, f32>,
    fused: &[(String, f32)],
    query: &str,
) -> Vec<SearchResult> {
    let mut map = HashMap::new();
    for doc in docs {
        map.insert(doc.doc_id.clone(), doc);
    }
    fused
        .iter()
        .filter_map(|(doc_id, fused_score)| {
            let doc = map.get(doc_id)?;
            Some(SearchResult {
                doc_id: doc.doc_id.clone(),
                concept_path: doc.concept_path.clone(),
                title: doc.title.clone(),
                type_name: doc.type_name.clone(),
                tags: doc.tags.clone(),
                snippet: snippet_from_doc(doc, query),
                bm25_score: bm25.get(doc_id).copied(),
                vector_score: vectors.get(doc_id).copied(),
                fused_score: *fused_score,
            })
        })
        .collect()
}
