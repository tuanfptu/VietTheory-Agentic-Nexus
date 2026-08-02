# MLN111 Structure and Parent-Child Chunking Gate

## Heading strategy

MLN111 uses a nearly uniform 13-point body font. Heading recognition therefore
combines PDF font flags with textbook numbering patterns instead of relying on font
size:

- level 1: `Chương N` plus an uppercase title;
- level 2: Roman numeral divisions and uppercase `A./B./C.` divisions;
- level 3: bold numbered sections;
- level 4: `a)/b)/...` subsections;
- level 5: explicit `*` micro-headings.

The parser ignores preface lists before the first chapter, stops heading detection at
`MỤC LỤC`, rejects citation prose such as `C. Mác...`, and attaches stacked heading
lines to their following content to avoid heading-only retrieval units.

## Structure artifact

- Headings: 231.
- Levels: 3 chapter, 19 division, 39 section, 77 subsection, 93 micro-heading.
- Detected chapter starts: PDF pages 7, 67 and 162.
- Invalid heading parent links: 0.

## Parent-child artifact

- Parents: 175.
- Children: 603.
- Parent token min/median/mean/max: 21 / 1251 / 987.6 / 1499.
- Child token min/median/mean/max: 21 / 385 / 330.6 / 400.
- Source body lines: 9,090.
- Parent source spans: 9,090, all unique.
- Section-boundary violations: 0.
- Children with missing parent: 0.
- Parent chunks above 1,500 tokens: 0.
- Child chunks above 400 tokens: 0.

The prior fixed-size baseline remains unchanged for ablation.

## Structured dense index

- Model: Qwen/Qwen3-Embedding-0.6B.
- Revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Index: normalized FAISS `IndexFlatIP`.
- Vectors: 603 x 1024.
- Hardware-selected batch size: 8.
- Child artifact SHA-256:
  `2636cf38617bdce6f937f584d736ad46399cb204e8fc751a2f5b4988b6e78773`.
- Index SHA-256:
  `b02edfabfcd7d52e6daf754ba37d0b4f57bd4f90c521a6136cd6125e0d30ecfa`.
- Mapping SHA-256:
  `ed72914d4924e586d86d64ed229f26851bd280d4657acf5d491af8434621aef0`.

## Retrieval and parent expansion smoke

For “Ý thức có nguồn gốc tự nhiên và xã hội như thế nào?”, rank 1 and 2 expanded
to parents under:

- Chapter 2;
- section `2. Nguồn gốc, bản chất và kết cấu của ý thức`;
- subsection `a) Nguồn gốc của ý thức`;
- PDF pages 87–90.

The expansion deduplicates siblings, preserves source spans, and bounds generation
context to the requested number of unique parents.

## Gate result

**PASS for structural correctness, complete provenance, parent-child constraints,
structured index integrity and qualitative parent-expansion retrieval.** Comparative
quality against fixed-size chunks remains pending the reviewed development benchmark.
