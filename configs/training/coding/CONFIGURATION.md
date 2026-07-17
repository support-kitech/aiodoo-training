# coding — Configuration Isolation

This directory is the sole public production configuration for the **coding**
training product.

```bash
python train.py --config configs/training/coding/experiment.yaml
```

All production knobs live here as sibling fragments. Shared templates under
`configs/training/default.yaml` (hyperparameter defaults) are not included by
this root — production values are explicit in the sibling YAML files.
