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

/// On-disk index format version.
pub const INDEX_FORMAT_VERSION: u32 = 3;
/// Magic bytes identifying a segment file.
pub const SEGMENT_MAGIC: &[u8; 8] = b"OKFSEG03";

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

/// Deserialized segment payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentFile {
    pub segment_id: String,
    pub metadata: SegmentMetadata,
    pub documents: Vec<OkfDocument>,
    pub bm25: Bm25Index,
    pub embeddings: Vec<Vec<f32>>,
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
    fn documents(&self) -> &[OkfDocument];
    fn bm25(&self) -> &Bm25Index;
    fn embeddings(&self) -> &[Vec<f32>];
}

impl SegmentReader for SegmentFile {
    fn metadata(&self) -> &SegmentMetadata {
        &self.metadata
    }
    fn documents(&self) -> &[OkfDocument] {
        &self.documents
    }
    fn bm25(&self) -> &Bm25Index {
        &self.bm25
    }
    fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }
}

/// File-backed storage for manifests and segments.
pub struct IndexStorage {
    root: PathBuf,
}

impl IndexStorage {
    /// Opens or creates the index root directory.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("segments"))?;
        Ok(Self { root })
    }

    /// Returns the manifest file path.
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
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
        let tmp_file = staging_dir.join("segment.bin.tmp");
        write_segment_bin(&tmp_file, segment)?;
        fs::rename(&tmp_file, staging_dir.join("segment.bin"))?;
        fs::rename(&staging_dir, &final_dir)?;
        Ok(SegmentEntry {
            segment_id: segment.segment_id.clone(),
            path: final_dir.to_string_lossy().to_string(),
            doc_count: segment.metadata.doc_count,
            created_at: segment.metadata.created_at,
        })
    }

    /// Loads a segment from disk.
    pub fn read_segment(&self, entry: &SegmentEntry) -> Result<SegmentFile> {
        read_segment_bin(Path::new(&entry.path).join("segment.bin"))
    }

    /// Returns the index root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[derive(Clone, Copy, Debug)]
struct Header {
    doc_count: u32,
    term_count: u32,
    posting_count: u32,
    vector_dim: u32,
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

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
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

    let header = Header {
        doc_count: segment.documents.len() as u32,
        term_count: terms.len() as u32,
        posting_count: (postings.len() / 8) as u32,
        vector_dim: segment.metadata.embedding_dimension as u32,
        docs_offset: 0,
        docs_len: (docs.len() * std::mem::size_of::<DocEntry>()) as u64,
        strings_offset: 0,
        strings_len: strings.len() as u64,
        terms_offset: 0,
        terms_len: (terms.len() * std::mem::size_of::<TermEntry>()) as u64,
        postings_offset: 0,
        postings_len: postings.len() as u64,
        vectors_offset: 0,
        vectors_len: vectors.len() as u64,
    };

    let header_size = 8 + 4 + 4 + 4 + 4 + 8 * 6;
    let docs_offset = header_size as u64;
    let strings_offset = docs_offset + header.docs_len;
    let terms_offset = strings_offset + header.strings_len;
    let postings_offset = terms_offset + header.terms_len;
    let vectors_offset = postings_offset + header.postings_len;
    let mut header = header;
    header.docs_offset = docs_offset;
    header.strings_offset = strings_offset;
    header.terms_offset = terms_offset;
    header.postings_offset = postings_offset;
    header.vectors_offset = vectors_offset;

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    file.write_all(SEGMENT_MAGIC)?;
    file.write_all(&INDEX_FORMAT_VERSION.to_le_bytes())?;
    file.write_all(&header.doc_count.to_le_bytes())?;
    file.write_all(&header.term_count.to_le_bytes())?;
    file.write_all(&header.posting_count.to_le_bytes())?;
    file.write_all(&header.vector_dim.to_le_bytes())?;
    for v in [
        header.docs_offset,
        header.docs_len,
        header.strings_offset,
        header.strings_len,
        header.terms_offset,
        header.terms_len,
        header.postings_offset,
        header.postings_len,
        header.vectors_offset,
        header.vectors_len,
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

fn write_u64_to(file: &mut File, v: u64) -> Result<()> {
    file.write_all(&v.to_le_bytes())?;
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

fn read_segment_bin(path: PathBuf) -> Result<SegmentFile> {
    let file = File::open(path)?;
    let mmap = map_file(&file)?;
    let mut cursor = 0usize;
    if mmap.get(0..8) != Some(SEGMENT_MAGIC.as_slice()) {
        return Err(anyhow!("invalid segment magic"));
    }
    cursor += 8;
    let version = read_u32(&mmap, &mut cursor)?;
    if version != INDEX_FORMAT_VERSION {
        return Err(anyhow!("unsupported segment version"));
    }
    let doc_count = read_u32(&mmap, &mut cursor)? as usize;
    let term_count = read_u32(&mmap, &mut cursor)? as usize;
    let _posting_count = read_u32(&mmap, &mut cursor)? as usize;
    let vector_dim = read_u32(&mmap, &mut cursor)? as usize;
    let header = Header {
        doc_count: doc_count as u32,
        term_count: term_count as u32,
        posting_count: _posting_count as u32,
        vector_dim: vector_dim as u32,
        docs_offset: read_u64(&mmap, &mut cursor)?,
        docs_len: read_u64(&mmap, &mut cursor)?,
        strings_offset: read_u64(&mmap, &mut cursor)?,
        strings_len: read_u64(&mmap, &mut cursor)?,
        terms_offset: read_u64(&mmap, &mut cursor)?,
        terms_len: read_u64(&mmap, &mut cursor)?,
        postings_offset: read_u64(&mmap, &mut cursor)?,
        postings_len: read_u64(&mmap, &mut cursor)?,
        vectors_offset: read_u64(&mmap, &mut cursor)?,
        vectors_len: read_u64(&mmap, &mut cursor)?,
    };

    let docs_bytes = mmap
        .get(header.docs_offset as usize..(header.docs_offset + header.docs_len) as usize)
        .ok_or_else(|| anyhow!("bad docs section"))?;
    let strings = mmap
        .get(header.strings_offset as usize..(header.strings_offset + header.strings_len) as usize)
        .ok_or_else(|| anyhow!("bad strings section"))?;
    let terms_bytes = mmap
        .get(header.terms_offset as usize..(header.terms_offset + header.terms_len) as usize)
        .ok_or_else(|| anyhow!("bad terms section"))?;
    let postings_bytes = mmap
        .get(
            header.postings_offset as usize
                ..(header.postings_offset + header.postings_len) as usize,
        )
        .ok_or_else(|| anyhow!("bad postings section"))?;
    let vectors_bytes = mmap
        .get(header.vectors_offset as usize..(header.vectors_offset + header.vectors_len) as usize)
        .ok_or_else(|| anyhow!("bad vectors section"))?;

    let mut docs = Vec::with_capacity(doc_count);
    let mut doc_cursor = 0usize;
    for _ in 0..doc_count {
        let entry = DocEntry {
            logical_key_off: read_u32(docs_bytes, &mut doc_cursor)?,
            logical_key_len: read_u32(docs_bytes, &mut doc_cursor)?,
            bundle_path_off: read_u32(docs_bytes, &mut doc_cursor)?,
            bundle_path_len: read_u32(docs_bytes, &mut doc_cursor)?,
            concept_path_off: read_u32(docs_bytes, &mut doc_cursor)?,
            concept_path_len: read_u32(docs_bytes, &mut doc_cursor)?,
            file_path_off: read_u32(docs_bytes, &mut doc_cursor)?,
            file_path_len: read_u32(docs_bytes, &mut doc_cursor)?,
            doc_id_off: read_u32(docs_bytes, &mut doc_cursor)?,
            doc_id_len: read_u32(docs_bytes, &mut doc_cursor)?,
            type_name_off: read_u32(docs_bytes, &mut doc_cursor)?,
            type_name_len: read_u32(docs_bytes, &mut doc_cursor)?,
            title_off: read_u32(docs_bytes, &mut doc_cursor)?,
            title_len: read_u32(docs_bytes, &mut doc_cursor)?,
            description_off: read_u32(docs_bytes, &mut doc_cursor)?,
            description_len: read_u32(docs_bytes, &mut doc_cursor)?,
            resource_off: read_u32(docs_bytes, &mut doc_cursor)?,
            resource_len: read_u32(docs_bytes, &mut doc_cursor)?,
            timestamp_off: read_u32(docs_bytes, &mut doc_cursor)?,
            timestamp_len: read_u32(docs_bytes, &mut doc_cursor)?,
            tags_off: read_u32(docs_bytes, &mut doc_cursor)?,
            tags_len: read_u32(docs_bytes, &mut doc_cursor)?,
            body_off: read_u32(docs_bytes, &mut doc_cursor)?,
            body_len: read_u32(docs_bytes, &mut doc_cursor)?,
            searchable_off: read_u32(docs_bytes, &mut doc_cursor)?,
            searchable_len: read_u32(docs_bytes, &mut doc_cursor)?,
        };
        let tags = read_string(strings, entry.tags_off, entry.tags_len)
            .map(|s| s.split('\n').map(str::to_string).collect())
            .unwrap_or_default();
        let doc = OkfDocument {
            logical_key: read_string(strings, entry.logical_key_off, entry.logical_key_len)
                .unwrap_or_default()
                .to_string(),
            bundle_path: read_string(strings, entry.bundle_path_off, entry.bundle_path_len)
                .unwrap_or_default()
                .to_string(),
            concept_path: read_string(strings, entry.concept_path_off, entry.concept_path_len)
                .unwrap_or_default()
                .to_string(),
            file_path: read_string(strings, entry.file_path_off, entry.file_path_len)
                .unwrap_or_default()
                .to_string(),
            doc_id: read_string(strings, entry.doc_id_off, entry.doc_id_len)
                .unwrap_or_default()
                .to_string(),
            type_name: read_string(strings, entry.type_name_off, entry.type_name_len)
                .unwrap_or_default()
                .to_string(),
            title: read_string(strings, entry.title_off, entry.title_len).map(str::to_string),
            description: read_string(strings, entry.description_off, entry.description_len)
                .map(str::to_string),
            resource: read_string(strings, entry.resource_off, entry.resource_len)
                .map(str::to_string),
            tags,
            timestamp: read_string(strings, entry.timestamp_off, entry.timestamp_len)
                .map(str::to_string),
            body: read_string(strings, entry.body_off, entry.body_len)
                .unwrap_or_default()
                .to_string(),
            searchable_text: read_string(strings, entry.searchable_off, entry.searchable_len)
                .unwrap_or_default()
                .to_string(),
        };
        docs.push(doc);
    }

    let mut bm25 = Bm25Index {
        config: Bm25Config::default(),
        doc_len: HashMap::new(),
        avg_doc_len: 0.0,
        doc_count,
        term_doc_freq: HashMap::new(),
        postings: HashMap::new(),
    };
    let mut term_cursor = 0usize;
    let mut total_len = 0usize;
    for _ in 0..term_count {
        let entry = TermEntry {
            term_off: read_u32(terms_bytes, &mut term_cursor)?,
            term_len: read_u32(terms_bytes, &mut term_cursor)?,
            postings_off: read_u32(terms_bytes, &mut term_cursor)? as u32,
            postings_len: read_u32(terms_bytes, &mut term_cursor)? as u32,
        };
        let term = read_string(strings, entry.term_off, entry.term_len)
            .unwrap_or_default()
            .to_string();
        let mut postings = HashMap::new();
        let start = entry.postings_off as usize;
        let end = (entry.postings_off + entry.postings_len) as usize;
        let slice = postings_bytes
            .get(start..end)
            .ok_or_else(|| anyhow!("bad postings slice"))?;
        let mut pcur = 0usize;
        while pcur < slice.len() {
            let doc_index = read_u32(slice, &mut pcur)? as usize;
            let tf = read_u32(slice, &mut pcur)? as usize;
            let doc_id = docs
                .get(doc_index)
                .ok_or_else(|| anyhow!("bad doc index"))?
                .doc_id
                .clone();
            *postings.entry(doc_id).or_insert(0) += tf;
        }
        bm25.term_doc_freq.insert(term.clone(), postings.len());
        bm25.postings.insert(term.clone(), postings);
    }

    for doc in &docs {
        let len = crate::bm25::tokenize(&doc.searchable_text).len();
        bm25.doc_len.insert(doc.doc_id.clone(), len);
        total_len += len;
    }
    bm25.avg_doc_len = if docs.is_empty() {
        0.0
    } else {
        total_len as f32 / docs.len() as f32
    };

    let mut embeddings = Vec::with_capacity(doc_count);
    let mut vcur = 0usize;
    for _ in 0..doc_count {
        let mut vec = Vec::with_capacity(vector_dim);
        for _ in 0..vector_dim {
            let bytes = vectors_bytes
                .get(vcur..vcur + 4)
                .ok_or_else(|| anyhow!("bad vector slice"))?;
            let array: [u8; 4] = bytes
                .try_into()
                .map_err(|_| anyhow!("invalid f32 encoding"))?;
            vec.push(f32::from_le_bytes(array));
            vcur += 4;
        }
        embeddings.push(vec);
    }

    Ok(SegmentFile {
        segment_id: "loaded".to_string(),
        metadata: SegmentMetadata {
            doc_count,
            avg_doc_len: bm25.avg_doc_len,
            embedding_dimension: vector_dim,
            created_at: 0,
        },
        documents: docs,
        bm25,
        embeddings,
    })
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

/// Maps a file into memory.
///
/// # Safety
///
/// The returned mapping is read-only and the file handle remains alive for the
/// duration of the map construction. The mapping is only used for immutable
/// reads of a segment file written by this crate.
fn map_file(file: &File) -> Result<Mmap> {
    // SAFETY: The mapping is read-only, the file is not mutated through this
    // handle, and the returned `Mmap` owns the mapping after construction.
    let mmap = unsafe { Mmap::map(file)? };
    Ok(mmap)
}
