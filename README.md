# Product-Segment-Analysis
## Objective

This project analyzes four years of sales data (2021–2024) to uncover product performance trends and customer purchase behavior using the FMC model (Frequency, Monetary, and Customer Variety).
The workflow integrates Python, Excel, and Tableau to perform a complete data analytics cycle:

Python → for data preparation, feature extraction and segmentation,

Excel → for consolidating yearly results,

Tableau → for interactive trend visualization and insight generation.

## Methodology

| Stage                           | Tool                             | Purpose                                                            |
| ------------------------------- | -------------------------------- | ------------------------------------------------------------------ |
| **1. Data Processing**          | 🐍 Python (Pandas, Scikit-learn) | Cleaning raw datasets, extracting FMC features, and product clustering per year         |
| **2. Result Aggregation**       | 📊 Microsoft Excel               | Merging all yearly outputs (2021–2024) for trend comparison        |
| **3. Visualization & Insights** | 📈 Tableau                       | Creating interactive dashboards to analyze multi-year FMC patterns |

## FMC Model Components

| Metric                   | Description                           | Business Meaning                                   |
| ------------------------ | ------------------------------------- | -------------------------------------------------- |
| **Frequency (F)**        | Number of transactions per product    | Product sales volume, a measure of product demand and popularity in units      |
| **Monetary (M)**         | Total revenue generated per product   | Indicates product profitability and sales strength |
| **Customer Variety (C)** | Count of unique customers per product | Reflects market reach and customer diversity       |

## Python Data Processing

All four datasets (sales_2021.csv – sales_2024.csv) were processed using the same Python script to ensure consistency.

📄 Script: fmc_sales_processing.py

Main steps inside the code:

Data Cleaning → remove duplicates, normalize text, handle missing values.

Feature Extraction → calculate Frequency, Monetary, and Customer Variety per product.

Normalization & Clustering → scale features and group products using K-Means ang Agglomerative Hierarchical Clustering

Evaluation → measure cluster quality using Silhouette Score.

Export Result → save each year’s FMC data as CSV in processed_results/

## Excel Integration

After processing, all yearly FMC CSV files were combined in Excel to create a single dataset for Tableau visualization.

The merged Excel file includes:

| Tahun	Barang	| Frequency |	Monetary	| Customer Variety	| K-Means Cluster | AHC Cluster |

💡 This manual consolidation step ensures full control and validation of data before visualization.
