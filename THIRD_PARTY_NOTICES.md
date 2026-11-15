# Third-party sources and licensing

- EfficientPhys implementations and derivatives retain author headers: Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff, WACV 2023. Upstream: https://github.com/ubicomplab/rPPG-Toolbox . The bundled `ops/deep_hr/TOOLBOX_LICENSE.txt` is the Responsible AI Source Code License 1.1; it is not MIT. Local variants modify normalization, shape handling, temporal shift boundary behavior and/or training integration. Consult each dated implementation and the experiment history.
- Some experiment snapshots include copies of the above implementation under `vendor/` or dated folders; the same upstream terms remain applicable. Copying a file into an experiment does not relicense it.
- LQ-rPPG source snapshots are retained only with their upstream license/notice files; this branch did not produce a validated zero-shot score. See its source README/license where included. It is not part of the supported quick start.
- POS, CHROM and PBV implementations implement published methods. Cite Wang et al., *Algorithmic Principles of Remote PPG* (POS); de Haan & Jeanne, *Robust Pulse Rate From Chrominance-Based rPPG* (CHROM); de Haan & van Leest, *Improved motion robustness ... blood volume pulse signature* (PBV), alongside implementation-specific notices.
- README organization was informed by rPPG-Toolbox and https://github.com/ZitongYu/PhysFormer; their README prose has not been copied.

No new blanket license is assigned to the team's original additions in this publication. Access to source does not override upstream licenses or dataset agreements. Third-party papers, organizer slide decks, original datasets, checkpoints, participant imagery and authentication helpers are not redistribution deliverables here.

Vendored visualization dependencies (`local_vendor/umap`, `pynndescent`) retain their distribution license files under the adjacent `*.dist-info/licenses/` directories. They are archival dependencies, not algorithms authored by this team.

The archived `sources/pyVHR/pyVHR/BVP/methods.py` is from https://github.com/phuselab/pyVHR and is accompanied by its GPL-3.0 license. It is an inspected upstream source snapshot, not a complete bundled pyVHR installation. LQ-rPPG source is from https://github.com/junseonglee-repo/LQ-rPPG; its bundled Mamba has its own adjacent license. User-supplied causal audit comparison utilities are retained under `causal_audit_20261113/supplied/` and `tool_validation/`; no new license is asserted for those supplied additions.
