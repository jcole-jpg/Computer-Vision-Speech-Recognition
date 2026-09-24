# Why MILK10k does not reflect real-world prevalence (B2)

**MILK10k skin-lesion classification** · Joshua Cole · companion to `milestone1_report.md`

---

MILK10k is **biopsy-enriched**, and the enrichment is measurable in its own metadata. A
lesion only enters the dataset if a clinician was suspicious enough to excise it: 95.7% of
lesions carry a histopathological label, and those lesions are **72.3% malignant**, against
**2.2%** for the 224 lesions confirmed by clinical assessment alone. The dataset's headline
69.4% malignancy rate is therefore a property of the *referral and biopsy funnel*, not of
skin. In unselected primary care the ratio is close to inverted — the overwhelming majority
of lesions a patient presents are benign, are recognised as benign on sight, and are never
biopsied, so they never appear here at all. What MILK10k samples is the hard residue left
after clinical triage has already removed the easy negatives.

This changes how every accuracy figure must be read. A metric computed here describes
performance **on lesions already judged worth excising**, which is a genuinely useful thing
to measure — it is roughly the specialist's decision problem — but it is not the GP's. Two
consequences follow. First, **precision will collapse under the real base rate even if
recall is unchanged**: with malignancy at ~5% rather than ~69%, the same sensitivity and
specificity produce far more false positives per true positive, because the negative pool
the model is wrong about is an order of magnitude larger. Second, **plain accuracy is
actively misleading**: predicting "Malignant" for everything scores 0.694 here and ~0.05 in
a clinic. A3.4 shows the ranking itself inverting — always-Malignant is the best constant
predictor by accuracy on this test set and the worst under a realistic prevalence, while
remaining the cheapest under the cost matrix in both.

The practical response is to report macro-averaged recall with per-class support, treat the
absolute numbers as an upper bound conditional on biopsy referral, and recalibrate
probabilities against local prevalence before any clinical use.

*(287 words)*
