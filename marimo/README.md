# RFGBoost Marimo Notebooks 🌊🌳

Interactive notebooks built with [Marimo](https://marimo.io) for exploring RFGBoost functionality.

## 📁 Available Notebooks

### 🔬 `marimo_notebook.py`
**Model dashboard** with comprehensive model analysis:
- 📊 Real bank dataset analysis (or synthetic fallback)
- 🎛️ Complete hyperparameter exploration
- 📈 Multi-metric performance tracking
- 📋 Overfitting detection

## 🏃 Quick Start

1. **Install dependencies**:
```bash
uv sync --dev  # Installs marimo and all dependencies
```

2. **Convert to IPython Notebook**:
```bash
uv run marimo export ipynb marimo/marimo_notebook.py -o marimo/marimo_notebook.ipynb
```

3. **Open in browser**: Marimo will automatically open the interactive notebook in your browser

```bash
uv run marimo edit marimo/marimo_notebook.py
```

## ✨ Features Demonstrated

### 🎛️ Interactive Controls
- **Dataset Configuration**: Sample size, random seeds
- **Model Parameters**: Base learners, boosting rounds, learning rates
- **Visualization Options**: Different chart types and metrics

### 📊 Real-time Analysis
- **Performance Metrics**: AUC, accuracy, log-loss, Brier score
- **Feature Analysis**: Importance rankings and WOE mappings
- **Model Diagnostics**: Overfitting detection and confidence intervals

### 🎯 Educational Value
- **Algorithm Understanding**: Step-by-step boosting process
- **Parameter Impact**: See how changes affect performance
- **Best Practices**: Demonstrated through interactive examples

## 🔧 Customization

### Adding New Notebooks
1. Create a new `.py` file in the `marimo/` folder
2. Use the marimo app structure:
   ```python
   import marimo
   app = marimo.App()
   
   @app.cell
   def your_cell():
       # Your code here
       return
   ```

### Extending Existing Notebooks
- **Add new datasets**: Modify the data loading sections
- **Include new metrics**: Extend the performance evaluation
- **Create visualizations**: Use Altair, Matplotlib, or Plotly

## 📚 Resources

- **Marimo Documentation**: https://docs.marimo.io
- **RFGBoost Repository**: https://github.com/xRiskLab/rfgboost
- **Altair Gallery**: https://altair-viz.github.io/gallery/

## 🎯 Benefits of Marimo Notebooks

Unlike Jupyter notebooks, Marimo notebooks are:
- ✅ **Reactive**: Automatic updates when parameters change
- ✅ **Reproducible**: No hidden state or execution order issues
- ✅ **Git-friendly**: Plain Python files, easy to version control
- ✅ **Interactive**: Rich UI components for parameter exploration
- ✅ **Fast**: Efficient execution and caching

## 🚀 Next Steps

1. **Start with `marimo_notebook.py`** to learn the basics
2. **Create your own notebook** for specific use cases
3. **Share your insights** with the RFGBoost community!
