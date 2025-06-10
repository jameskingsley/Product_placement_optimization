# Product_placement_optimization

# Market Basket Analysis with Apriori (Product Placement Optimization)

This Streamlit application allows you to perform market basket analysis using the Apriori algorithm. The app helps to discover frequent itemsets, generate association rules, and visualize the relationships between products based on their co-occurrence in transactions. It is designed to assist in finding patterns for product placement, bundle suggestions, and promotional strategies.

## Features

- **Dataset Upload**: Upload a cleaned dataset in CSV format (e.g., transactional data).
- **Basket Matrix Creation**: Converts the dataset into a basket matrix (items vs. transactions).
- **Apriori Algorithm**: Runs the Apriori algorithm to find frequent itemsets.
- **Association Rules**: Generates association rules with metrics like support, confidence, and lift.
- **Visualization**:
  - Scatter plot: Support vs. Confidence (colored by Lift)
  - Network graph: Visualizes association rules with nodes (products) and edges (relationships)
- **Product Filtering**: Filter and display rules related to a specific product (antecedent or consequent).

