import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx

st.title("Retail Product Placement Optimization with Apriori(Market Basket Analysis)")

# Upload or load dataset
uploaded_file = st.file_uploader("Upload your cleaned dataset CSV file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df  
    #Save dataframe to session state
else:
    if 'df' not in st.session_state:
        st.warning("Please upload a cleaned dataset CSV file.")
        st.stop()
    df = st.session_state.df  
    #Retrieve dataframe from session state

#Show basic data info
st.write("## Dataset Sample")
st.dataframe(df.head())

#Creating basket matrix
st.write("## Creating Basket Matrix...")
df['Quantity'] = df['Quantity'].astype(int)
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

st.write("Basket matrix shape:", basket_sets.shape)

# Apriori parameters
st.write("## Apriori Parameters")
min_support = st.slider("Minimum Support", min_value=0.01, max_value=0.1, value=0.02, step=0.01)
min_confidence = st.slider("Minimum Confidence for Rules", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
min_lift = st.slider("Minimum Lift for Rules", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

# Running Apriori
if 'frequent_itemsets' not in st.session_state:
    if st.button("Run Apriori and Generate Rules"):
        with st.spinner("Generating frequent itemsets..."):
            frequent_itemsets = apriori(basket_sets.astype(bool), min_support=min_support, use_colnames=True)
            st.session_state.frequent_itemsets = frequent_itemsets  # Save to session state
            st.write(f"Found {len(frequent_itemsets)} frequent itemsets")
            st.dataframe(frequent_itemsets.sort_values(by='support', ascending=False).head(10))
        
        with st.spinner("Generating association rules..."):
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            rules = rules[rules['lift'] >= min_lift]
            st.session_state.rules = rules  # Save to session state
            st.write(f"Found {len(rules)} association rules after filtering")
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
else:
    frequent_itemsets = st.session_state.frequent_itemsets
    rules = st.session_state.rules

# Checking if 'rules' exist before proceeding
if 'rules' in st.session_state:
    rules = st.session_state.rules
    
    #Visualization: Scatter plot
    st.write("## Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Lift')
    ax.set_xlabel('Support')
    ax.set_ylabel('Confidence')
    ax.set_title('Support vs Confidence (colored by Lift)')
    st.pyplot(fig)

    #Network graph for top 20 rules
    st.write("### Association Rules Network Graph (Top 20 rules)")
    G = nx.DiGraph()
    top_rules = rules.head(20)
    for _, row in top_rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'])

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color=weights,
            width=3, edge_cmap=plt.cm.plasma, node_size=2500, font_size=10)
    st.pyplot(plt.gcf())
    plt.clf()

#Filtering rules by product input
st.write("## Filter Rules by Product")
product_input = st.text_input("Enter product name to filter rules (antecedent or consequent)")
if product_input and 'rules' in st.session_state:
    rules = st.session_state.rules

    #Converting frozensets to strings for matching
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    filtered_rules = rules[
        rules['antecedents_str'].str.contains(product_input, case=False) |
        rules['consequents_str'].str.contains(product_input, case=False)
    ]
    
    st.write(f"Filtered rules containing '{product_input}': {len(filtered_rules)}")
    st.dataframe(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Business insights summary
st.write("## Business Insights Summary")
st.markdown("""
- Place frequently associated products close together in your store.
- Create bundles or combo discounts for highly associated products.
- Use promotions targeting customers who buy certain products to upsell related items.
- Stock associated products together and monitor inventory to avoid stockouts.
""")
