# Model weights

The weights are archived on Zenodo (see the main README for the DOI), not in this repository. Each model is an ensemble of five fold models; a photograph's estimate is the geometric mean of the five predicted masses.


## Nutrition5k-only DINOv2-base

Training configuration: `stage_b_imageonly_aux10/config.json`.

| File | Size (MB) | SHA-256 |
|---|---|---|
| `stage_b_imageonly_aux10/fold0.pt` | 350 | `09b125589056144b41a9df821707a547ca7ba62c77aed48110883a5738d6683f` |
| `stage_b_imageonly_aux10/fold1.pt` | 350 | `a47776e0bdd5947632dad4f8666782d9c20697f3a5c1832f3cd8372c8b5c4201` |
| `stage_b_imageonly_aux10/fold2.pt` | 350 | `54ac43e9daf30554f62ae6945ff96dc846f14a7374c00a4bf62da7451218fa42` |
| `stage_b_imageonly_aux10/fold3.pt` | 350 | `8dc1f2d39a8cbac5e51ac2eb3ed764b7e08e2514f1e5dbf2c433fb26799661f4` |
| `stage_b_imageonly_aux10/fold4.pt` | 350 | `998de81f5674a908648cf1b0655969f2064ef15c77284c98a1fd9b33d67d7735` |

## Two-domain DINOv2-large (Nutrition5k + expanded ASA24)

Training configuration: `joint_large_consensus/config.json`.

| File | Size (MB) | SHA-256 |
|---|---|---|
| `joint_large_consensus/fold0.pt` | 1222 | `8959043ca39977a2f2f5dd27321358470af8d444a213c025d8b25f994fa8742b` |
| `joint_large_consensus/fold1.pt` | 1222 | `67ed41234b9dec3e6ad4543bf48badafbaee2ae1b66302e257120d5bb09c4411` |
| `joint_large_consensus/fold2.pt` | 1222 | `c6eda085fcfa8a79d2a0cdb62a3f7c37a904a3f773aec2bcc1eab1d71ccae9e2` |
| `joint_large_consensus/fold3.pt` | 1222 | `de5fff41d2352d69bc0d4d6b30a0a3ff959de4cde1d2da2f61a5cdf79005190f` |
| `joint_large_consensus/fold4.pt` | 1222 | `330534471bf0c30078cd9ea53102ec2948dfc73b517b68c4ee5957168560f918` |

## Three-domain DINOv2-base (Nutrition5k + ASA24 + FPB)

Training configuration: `joint3_base/config.json`.

| File | Size (MB) | SHA-256 |
|---|---|---|
| `joint3_base/fold0.pt` | 353 | `36ae81039ba6674305a0abc339408e21701673f0ce44d4e88395d4588d544780` |
| `joint3_base/fold1.pt` | 353 | `fcb696766e513c5ebff6ca4d8277f7fd1473477bdf307ef9d5fc94de6d7566b8` |
| `joint3_base/fold2.pt` | 353 | `582609051e2897f91ce9874f9198a82834010b5409674e1fb9b18b747f6c78cb` |
| `joint3_base/fold3.pt` | 353 | `bd3598a001647b145002818c00a0345d055910725aba494441f7866379182fff` |
| `joint3_base/fold4.pt` | 353 | `23f5ab472ac74c8f8d8dc1e9a514caa3f4cbbe3a0fd9db055258f8b06bd5d1d2` |
