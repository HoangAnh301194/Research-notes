# IEEE Access manuscript (working draft)

Entry point: `main.tex`.

## Build

1. Download the official IEEE Access LaTeX template from https://ieeeaccess.ieee.org/authors/preparing-your-article/ or open https://www.overleaf.com/latex/templates/ieee-access-latex-template/cdxrhtbjgszv .
2. Copy the official template class `ieeeaccess.cls` and its companion resources into this directory (or upload this manuscript into the official Overleaf template). Do not substitute the generic IEEEtran class.
3. Open `Figures/VideoMamba_Technical_Diagrams_v3.drawio.html` in diagrams.net, edit the embedded diagram, and export pages 1--3 individually as `Figures/Fig1.pdf`, `Figures/Fig2.pdf`, and `Figures/Fig3.pdf`. PNG with the same base names also works for drafting. The manuscript uses explicitly labeled placeholders until the graphics are exported.
4. From this directory, run `pdflatex main.tex` twice, or select pdfLaTeX on Overleaf. The current references use an inline thebibliography environment (no BibTeX run needed).

## Draft scope

Sections III-A through III-C describe the implemented offline RGB recognition pipeline in the IPN-Hand and CTPGesture P1 notebooks. Section III-D documents scope and computation, not a novel adaptive algorithm. Figure 4 in the HTML is a research concept and is intentionally excluded from the manuscript until implementation and evaluation. Author names, affiliations, DOI, final results, and funding must be completed before submission.

Implementation references: ../KaggleNotebook/VideoMamba_IPN_Hand_P1_Metrics_Keyframes_Python313.ipynb and ../KaggleNotebook/VideoMamba_CTPGesture_P1_Metrics_Keyframes_Python313.ipynb.
