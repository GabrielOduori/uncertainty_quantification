"""
PDF Content Extraction for Literature Review

This module provides tools to extract specific content from PDFs
for building justification and literature review sections.

Author: Gabriel Oduori
Date: 2026-01-02
"""

from pathlib import Path
from typing import Dict, List, Optional
import json


class KeyPaperExtractor:
    """
    Extract key information from important uncertainty quantification papers.
    """

    def __init__(self, literature_dir: str):
        """
        Initialize extractor.

        Parameters
        ----------
        literature_dir : str
            Path to literature directory
        """
        self.literature_dir = Path(literature_dir)
        self.key_papers = self._identify_key_papers()

    def _identify_key_papers(self) -> Dict[str, Dict]:
        """
        Identify key papers that should be prioritized for extraction.

        Returns
        -------
        Dict[str, Dict]
            Mapping of paper type to file information
        """
        return {
            'kiureghian': {
                'filename': 'derkiureghian_paper.pdf',
                'title': 'Aleatory or epistemic? Does it matter?',
                'relevance': 'Foundational taxonomy of uncertainty types',
                'key_sections': [
                    'Introduction to epistemic vs aleatoric',
                    'Mathematical formalization',
                    'Engineering applications'
                ],
                'extract_for': ['Introduction', 'Theoretical Foundation']
            },
            'malings2024': {
                'filename': 'Journal of Geophysical Research  Machine Learning and Computation - 2024 - Malings - Air Quality Estimation and Forecasting.pdf',
                'title': 'Air Quality Estimation and Forecasting With Uncertainty',
                'relevance': 'State-of-art air quality UQ with multi-source fusion',
                'key_sections': [
                    'Multi-source data fusion approach',
                    'Uncertainty quantification methodology',
                    'Calibration evaluation',
                    'Results and findings'
                ],
                'extract_for': ['Literature Review', 'Methodology', 'Comparison']
            },
            'gal2016': {
                'filename': 'Uncertainty in Deep Learning.pdf',
                'title': 'Uncertainty in Deep Learning (PhD Thesis)',
                'relevance': 'MC Dropout, Bayesian deep learning, epistemic/aleatoric',
                'key_sections': [
                    'Uncertainty representation in neural networks',
                    'MC Dropout methodology',
                    'Epistemic vs aleatoric decomposition',
                    'Calibration techniques'
                ],
                'extract_for': ['Related Work', 'Methodology Comparison']
            },
            'li2017': {
                'filename': 'Geophysical Research Letters - 2017 - Li - Estimating Ground‐Level PM2 5 by Fusing Satellite and Station Observations  A.pdf',
                'title': 'Estimating Ground-Level PM2.5 by Fusing Satellite and Station Observations',
                'relevance': 'Satellite + ground monitor fusion for air quality',
                'key_sections': [
                    'Multi-source fusion methodology',
                    'Uncertainty treatment',
                    'Spatial prediction results'
                ],
                'extract_for': ['Literature Review', 'Applications']
            },
            'lindley2006': {
                'filename': 'Lindley-D.V.-Understanding-uncertainty-2006.pdf',
                'title': 'Understanding Uncertainty',
                'relevance': 'Philosophical foundations of probability and uncertainty',
                'key_sections': [
                    'Bayesian interpretation of probability',
                    'Uncertainty vs variability',
                    'Decision-making under uncertainty'
                ],
                'extract_for': ['Introduction', 'Philosophical Context']
            },
            'elicit': {
                'filename': 'Elicit - Effective Uncertainty Quantification in Low-Cost S - Report.pdf',
                'title': 'Effective Uncertainty Quantification in Low-Cost Sensors',
                'relevance': 'Low-cost sensor uncertainty specific',
                'key_sections': [
                    'Low-cost sensor error characteristics',
                    'Calibration uncertainty',
                    'Uncertainty propagation'
                ],
                'extract_for': ['Data Sources', 'Methodology']
            }
        }

    def generate_extraction_prompts(self) -> Dict[str, List[str]]:
        """
        Generate prompts for extracting information from each key paper.

        Returns
        -------
        Dict[str, List[str]]
            Mapping of paper key to list of extraction prompts
        """
        prompts = {
            'kiureghian': [
                'What is the formal definition of epistemic uncertainty?',
                'What is the formal definition of aleatoric uncertainty?',
                'What are the key differences between the two types?',
                'What mathematical framework is proposed for decomposition?',
                'What are practical implications for engineering systems?'
            ],
            'malings2024': [
                'What multi-source data fusion approach is used?',
                'How is uncertainty quantified in the model?',
                'What calibration metrics are used?',
                'What are the key findings about uncertainty in air quality forecasting?',
                'How do different data sources contribute to uncertainty?',
                'What gaps or future work are identified?'
            ],
            'gal2016': [
                'How is epistemic uncertainty represented in neural networks?',
                'How is aleatoric uncertainty represented?',
                'What is the MC Dropout approach?',
                'How is uncertainty decomposition performed?',
                'What calibration techniques are discussed?',
                'What are the limitations of deep learning UQ approaches?'
            ],
            'li2017': [
                'How are satellite and ground observations fused?',
                'How is spatial uncertainty handled?',
                'What uncertainty estimates are provided?',
                'What are the main results for PM2.5 estimation?',
                'What are the identified limitations?'
            ],
            'lindley2006': [
                'What is the Bayesian interpretation of probability?',
                'How does uncertainty differ from variability?',
                'What role does uncertainty play in decision-making?',
                'What are the philosophical foundations?'
            ],
            'elicit': [
                'What are the main sources of uncertainty in low-cost sensors?',
                'How does calibration uncertainty manifest?',
                'What methods are proposed for UQ in low-cost sensors?',
                'What are the recommended best practices?'
            ]
        }

        return prompts

    def create_extraction_plan(self, output_file: str) -> None:
        """
        Create a structured plan for extracting information from PDFs.

        Parameters
        ----------
        output_file : str
            Path to save extraction plan
        """
        plan = {
            'extraction_plan': {
                'description': 'Systematic plan to extract UQ information from key papers',
                'papers': []
            }
        }

        prompts = self.generate_extraction_prompts()

        for paper_key, paper_info in self.key_papers.items():
            paper_plan = {
                'key': paper_key,
                'filename': paper_info['filename'],
                'title': paper_info['title'],
                'relevance': paper_info['relevance'],
                'priority': self._get_priority(paper_key),
                'extraction_questions': prompts.get(paper_key, []),
                'use_in_sections': paper_info['extract_for'],
                'status': 'pending'
            }
            plan['extraction_plan']['papers'].append(paper_plan)

        # Save plan
        with open(output_file, 'w') as f:
            json.dump(plan, f, indent=2)

        print(f"Extraction plan saved to: {output_file}")
        print(f"Total papers to process: {len(plan['extraction_plan']['papers'])}")

    def _get_priority(self, paper_key: str) -> int:
        """
        Get priority ranking for paper extraction.

        Parameters
        ----------
        paper_key : str
            Paper identifier

        Returns
        -------
        int
            Priority (1=highest, 5=lowest)
        """
        high_priority = ['kiureghian', 'malings2024']
        medium_priority = ['gal2016', 'li2017']

        if paper_key in high_priority:
            return 1
        elif paper_key in medium_priority:
            return 2
        else:
            return 3

    def generate_literature_review_outline(self) -> str:
        """
        Generate an outline for the literature review section.

        Returns
        -------
        str
            Markdown formatted outline
        """
        outline = """
# Literature Review Outline

## 1. Introduction to Uncertainty Quantification (2-3 pages)

### 1.1 Fundamental Concepts
- **Der Kiureghian & Ditlevsen (2009)**: Epistemic vs aleatoric taxonomy
- **Lindley (2006)**: Bayesian interpretation of uncertainty
- Historical context and evolution

### 1.2 Importance in Environmental Modeling
- Public health decision-making
- Risk assessment and management
- Model validation and verification

**Key Papers to Extract From:**
- `derkiureghian_paper.pdf`
- `Lindley-D.V.-Understanding-uncertainty-2006.pdf`

---

## 2. Theoretical Foundations (3-4 pages)

### 2.1 Gaussian Process Uncertainty
- Predictive distributions
- Variance decomposition
- Hyperparameter uncertainty
- **Citations**: Rasmussen & Williams, relevant GP papers

### 2.2 Calibration Theory
- Proper scoring rules (Gneiting & Raftery, 2007)
- Coverage probability
- Calibration error metrics
- **Citations**: Guo et al. (2017) for neural network calibration

### 2.3 Epistemic-Aleatoric Decomposition
- Mathematical formalization
- Practical computation
- Interpretation guidelines

**Key Papers to Extract From:**
- `derkiureghian_paper.pdf` (decomposition theory)
- Statistical theory papers

---

## 3. UQ in Machine Learning (2-3 pages)

### 3.1 Deep Learning Approaches
- **Gal (2016)**: MC Dropout, Bayesian neural networks
- Ensemble methods
- Calibration challenges

### 3.2 Gaussian Process Methods
- Exact GP uncertainty
- Sparse variational GPs
- Computational considerations

### 3.3 Hybrid Approaches
- GAM with uncertainty
- State space models
- Additive models

**Key Papers to Extract From:**
- `Uncertainty in Deep Learning.pdf`
- Relevant GP literature

---

## 4. Air Quality Specific Applications (3-4 pages)

### 4.1 Multi-Source Data Fusion
- **Malings et al. (2024)**: State-of-art forecasting with UQ
- **Li et al. (2017)**: Satellite + ground fusion
- **Google Air Quality Project**: Large-scale deployment

### 4.2 Sensor Uncertainty Characteristics
- EPA regulatory monitors: ±2 μg/m³
- Low-cost sensors: ±5-10 μg/m³ (Elicit report)
- Satellite retrievals: AOD conversion uncertainty

### 4.3 Existing UQ Approaches
- Kriging-based methods
- Ensemble forecasting
- Limitations and gaps

**Key Papers to Extract From:**
- `Journal of Geophysical Research [...] Malings [...].pdf`
- `Geophysical Research Letters - 2017 - Li [...].pdf`
- `Elicit - Effective Uncertainty [...].pdf`

---

## 5. Research Gaps and Motivation (2-3 pages)

### 5.1 Incomplete Uncertainty Quantification
- Most models: only total uncertainty
- Missing: epistemic/aleatoric decomposition
- Missing: hyperparameter uncertainty

### 5.2 Limited Calibration Evaluation
- Often report variance without validation
- Rarely evaluate PICP, ECE, CRPS together
- No comparative analysis across models

### 5.3 OOD Detection Absent
- Spatial extrapolation undetected
- Temporal drift unmonitored
- Overconfident predictions

### 5.4 Transfer Learning UQ Ignored
- Focus on predictive accuracy only
- Uncertainty transfer not characterized
- Calibration preservation not studied

**Justification for This Work:**
This research addresses all four gaps through:
1. Full epistemic/aleatoric decomposition (Chapter 7)
2. Comprehensive calibration evaluation (Chapter 9)
3. Automated OOD detection (Chapter 9)
4. Transfer learning UQ analysis (Chapter 10)

---

## 6. Methodology Selection Justification (2 pages)

### 6.1 Why Gaussian Processes?
- Natural uncertainty quantification
- Theoretical grounding
- Spatial modeling capabilities
- Well-calibrated when properly specified

### 6.2 Why Epistemic-Aleatoric Decomposition?
- Scientific interpretability
- Active learning guidance
- Transfer learning insights
- Decision-making utility

### 6.3 Why PICP, ECE, CRPS?
- Complementary metrics
- Standard in forecasting
- Proper scoring rules
- Stakeholder-interpretable

---

## Total Length: ~15-20 pages

## Estimated Figures: 3-5
1. Uncertainty taxonomy diagram (Der Kiureghian)
2. Calibration metrics comparison table
3. Air quality data sources uncertainty characteristics
4. Research gaps visualization
5. Methodology selection decision tree

---

## Next Steps for Extraction

1. **Priority 1 (This week):**
   - Extract from Malings et al. (2024)
   - Extract from Der Kiureghian (2009)

2. **Priority 2 (Next week):**
   - Extract from Gal (2016)
   - Extract from Li et al. (2017)
   - Extract from Elicit report

3. **Priority 3 (Following week):**
   - Extract from other supporting papers
   - Compile full bibliography
   - Draft integrated literature review
"""
        return outline.strip()


def main():
    """
    Main function to create extraction plan and outline.
    """
    lit_dir = "/media/gabriel-oduori/SERVER/dev_space/uncertainty_quantification/literature"
    docs_dir = "/media/gabriel-oduori/SERVER/dev_space/uncertainty_quantification/docs"

    # Initialize extractor
    extractor = KeyPaperExtractor(lit_dir)

    # Create extraction plan
    plan_file = f"{docs_dir}/extraction_plan.json"
    extractor.create_extraction_plan(plan_file)

    # Generate outline
    outline = extractor.generate_literature_review_outline()
    outline_file = f"{docs_dir}/literature_review_outline.md"

    with open(outline_file, 'w') as f:
        f.write(outline)

    print(f"\nLiterature review outline saved to: {outline_file}")

    # Print priority papers
    print("\n" + "="*80)
    print("PRIORITY PAPERS FOR EXTRACTION")
    print("="*80)

    for paper_key, paper_info in extractor.key_papers.items():
        priority = extractor._get_priority(paper_key)
        priority_label = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}[priority]
        print(f"\n[{priority_label}] {paper_key}")
        print(f"  File: {paper_info['filename']}")
        print(f"  Title: {paper_info['title']}")
        print(f"  Relevance: {paper_info['relevance']}")

    print("\n" + "="*80)
    print("NEXT ACTIONS")
    print("="*80)
    print("\n1. Use the Read tool to extract content from each PDF:")
    print("   Example: Read('literature/derkiureghian_paper.pdf')")
    print("\n2. Answer extraction questions for each paper (see extraction_plan.json)")
    print("\n3. Build literature review sections using extracted content")
    print("\n4. Generate BibTeX entries for all cited papers")


if __name__ == "__main__":
    main()
