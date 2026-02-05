# JSW Regression Interpretability (EigenCAM)

We use EigenCAM to visualize which parts of a knee joint mask a ResNet-50 model uses to predict specific **Joint Space Width (JSW)** measurements.

## Understanding `explain_target_idx`

The model predicts a vector of JSW values. `explain_target_idx` selects which specific measurement point to explain.

### Medial Compartment (7 Outputs)
| Index | Column | Anatomical Position |
| :--- | :--- | :--- |
| **0** | `V00JSW150` | Inner Edge |
| **3** | `V00JSW225` | **Joint Center** |
| **6** | `V00JSW300` | Outer Edge |


### Lateral Compartment (9 Outputs)
| Index | Column | Anatomical Position |
| :--- | :--- | :--- |
| **0** | `V00LJSW700` | Inner Edge |
| **4** | `V00LJSW800` | **Joint Center** |
| **8** | `V00LJSW900` | Outer Edge |

---

## Setup

### 1. Configuration (`regression_explain.yaml`)
Update your paths and choose the target index:
```yaml
test_csv: "/path/to/jsw_measurements.csv"
mask_dir: "/path/to/masks"
image_dir: "/path/to/images_side"
explain_target_idx: 3  # Explains the middle of the joint