# MolPipeline Chemistry Benchmark Tasks

Этот набор нужен для проверки системы на реальных заданиях с определенным ответом. Задания не сгенерированы: PDF-источники сохранены локально в `validation_materials/sources/`.

## Local PDF Sources

1. `validation_materials/sources/school_ncert_class12_unit13_amines.pdf`
   - Source: https://www.ncert.nic.in/pdf/publication/exemplarproblem/classXII/chemistry/leep513.pdf
   - Level: school, Class XII
   - Topic: Amines

2. `validation_materials/sources/school_ncert_class12_unit12_aldehydes_ketones_carboxylic_acids.pdf`
   - Source: https://www.ncert.nic.in/pdf/publication/exemplarproblem/classXII/chemistry/leep512.pdf
   - Level: school, Class XII
   - Topic: Aldehydes, ketones and carboxylic acids

3. `validation_materials/sources/university_openstax_organic_chemistry_10e_study_guide.pdf`
   - Source: https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry10e-StudyGuide_WIdXLUO.pdf
   - Level: university
   - Topic: Organic chemistry review units

4. `validation_materials/sources/research_segler_2018_nature_supplementary_information.pdf`
   - Source: https://static-content.springer.com/esm/art%3A10.1038%2Fnature25978/MediaObjects/41586_2018_BFnature25978_MOESM1_ESM.pdf
   - Related article: https://www.nature.com/articles/nature25978
   - Level: research
   - Topic: neural-symbolic retrosynthesis planning, MCTS vs BFS, route diversity

## School-Level Tasks

### S1 - Classify a tertiary amine

- Source PDF: `validation_materials/sources/school_ncert_class12_unit13_amines.pdf`
- PDF page with task: page 1
- Task location: Unit 13, `I. Multiple Choice Questions (Type-I)`, question 1
- PDF page with answer key: page 15
- Answer key location: `ANSWERS`, `I. Multiple Choice Questions (Type-I)`, answer 1
- Prompt:
  - `NCERT Class XII Amines: Which option is a tertiary amine: 1-methylcyclohexylamine, triethylamine, tert-butylamine, or N-methylaniline? Answer only with the correct option and compound.`
- Expected answer:
  - `(ii) Triethylamine`

### S2 - Rank amine basicity in aqueous medium

- Source PDF: `validation_materials/sources/school_ncert_class12_unit13_amines.pdf`
- PDF page with task: page 1
- Task location: Unit 13, `I. Multiple Choice Questions (Type-I)`, question 3
- PDF page with answer key: page 15
- Answer key location: `ANSWERS`, `I. Multiple Choice Questions (Type-I)`, answer 3
- Prompt:
  - `NCERT Class XII Amines: Among CH3NH2, NCCH2NH2, (CH3)2NH, and C6H5NHCH3, which is the strongest base in aqueous medium? Answer only with the correct option and compound.`
- Expected answer:
  - `(iii) (CH3)2NH`

### S3 - Rank acid strength

- Source PDF: `validation_materials/sources/school_ncert_class12_unit12_aldehydes_ketones_carboxylic_acids.pdf`
- PDF page with task: page 1
- Task location: Unit 12, `I. Multiple Choice Questions (Type-I)`, question 3
- PDF page with answer key: page 10
- Answer key location: `ANSWERS`, `I. Multiple Choice Questions (Type-I)`, answer 3
- Prompt:
  - `NCERT Class XII Aldehydes/Ketones/Carboxylic Acids: Choose the correct increasing acidic strength order among ethanol, phenol, acetic acid, and chloroacetic acid. Answer only with the correct option and order.`
- Expected answer:
  - `(iii) Ethanol < Phenol < Acetic acid < Chloroacetic acid`

## University-Level Tasks

### U1 - Carbonyl alpha chemistry: enol vs enolate

- Source PDF: `validation_materials/sources/university_openstax_organic_chemistry_10e_study_guide.pdf`
- PDF page with task: page 129
- Task location: `Review Unit 9: Carbonyl Compounds II - Reaction at the alpha Carbon; Amines`, `Multiple Choice`, question 2
- PDF page with answer key: page 222
- Answer key location: `Appendix I: Answers to Multiple Choice Questions in Review Units 1-12`, `Review Unit 9`, answer 2
- Prompt:
  - `OpenStax Organic Chemistry Review Unit 9: In which reaction is an enol, rather than an enolate, the reacting species: acetoacetic acid synthesis, malonic ester synthesis, LDA alkylation, or Hell-Volhard-Zelinskii reaction? Answer only with the correct option and reaction.`
- Expected answer:
  - `(d) Hell-Volhard-Zelinskii reaction`

### U2 - Identify reaction forming cyclohexenone

- Source PDF: `validation_materials/sources/university_openstax_organic_chemistry_10e_study_guide.pdf`
- PDF page with task: page 129
- Task location: `Review Unit 9`, `Multiple Choice`, question 6
- PDF page with answer key: page 222
- Answer key location: `Appendix I`, `Review Unit 9`, answer 6
- Prompt:
  - `OpenStax Organic Chemistry Review Unit 9: Which reaction forms a cyclohexenone: Dieckmann cyclization, Michael reaction, Claisen condensation, or intramolecular aldol condensation? Answer only with the correct option and reaction.`
- Expected answer:
  - `(d) intramolecular aldol condensation`

### U3 - Diazonium coupling to azo compound

- Source PDF: `validation_materials/sources/university_openstax_organic_chemistry_10e_study_guide.pdf`
- PDF page with task: page 130
- Task location: `Review Unit 9`, `Multiple Choice`, question 10
- PDF page with answer key: page 222
- Answer key location: `Appendix I`, `Review Unit 9`, answer 10
- Prompt:
  - `OpenStax Organic Chemistry Review Unit 9: To form an azo compound, an aryldiazonium salt should react with CuCN, benzene, nitrobenzene, or phenol? Answer only with the correct option and reagent.`
- Expected answer:
  - `(d) phenol`

## Research-Level Tasks

### R1 - Route diversity count in neural retrosynthesis planning

- Source PDF: `validation_materials/sources/research_segler_2018_nature_supplementary_information.pdf`
- PDF page with task/answer: page 3
- Task location: section `2 Analysing Route Diversity`
- Prompt:
  - `Segler, Preuss and Waller Nature 2018 supplementary information, section Analysing Route Diversity: how many additional routes were retrieved besides the route with highest score? Answer with the exact count and phrase.`
- Expected answer:
  - `19 additional routes`
- Why this fits MolPipeline:
  - It checks whether the system can use literature/web evidence about retrosynthesis planning rather than only solve textbook chemistry.

### R2 - 3N-MCTS route-search time budget

- Source PDF: `validation_materials/sources/research_segler_2018_nature_supplementary_information.pdf`
- PDF page with task/answer: page 3
- Task location: section `2 Analysing Route Diversity`
- Prompt:
  - `Segler, Preuss and Waller Nature 2018 supplementary information, section Analysing Route Diversity: what total simulation time was allocated for 3N-MCTS across all 20 routes? Answer with the exact time phrase.`
- Expected answer:
  - `600 s total simulation time`
- Why this fits MolPipeline:
  - It targets an implementation/evaluation detail of a retrosynthesis-search method, which is difficult to answer reliably without source lookup.

### R3 - MCTS vs BFS speed comparison

- Source PDF: `validation_materials/sources/research_segler_2018_nature_supplementary_information.pdf`
- PDF page with task/answer: page 4
- Task location: section `2.2 Results and Discussion`
- Prompt:
  - `Segler, Preuss and Waller Nature 2018 supplementary information, section 2.2 Results and Discussion: compared with BFS, how much faster was MCTS reported to be? Answer with the exact magnitude phrase.`
- Expected answer:
  - `two orders of magnitude faster`
- Why this fits MolPipeline:
  - It is directly about our product direction: route search quality and performance for retrosynthesis planning.

## Recommended Evaluation Rule

Do not grade by "close enough" prose. Grade only if the response includes `expected_contains` after normalization. This keeps the benchmark deterministic and avoids post-hoc interpretation.
