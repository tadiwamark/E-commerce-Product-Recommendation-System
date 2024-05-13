# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Function to load and preprocess data
def load_data(file_path):
    # Load the transaction data
    data = pd.read_csv(file_path)
    # Assuming the transaction data needs to be transformed for analysis
    # Typically, the data will need to be transformed to a format suitable for market basket analysis:
    # One-hot encoding or similar transformations can be performed here.
    basket = (data.groupby(['CustomerID', 'StockCode'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('CustomerID'))
    # Convert positive values to 1 and everything else to 0
    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1
    basket_sets = basket.applymap(encode_units)
    return basket_sets

# Function to apply FP-Growth and find frequent item sets and generate rules
def apply_fp_growth(basket_sets, min_support=0.01, min_confidence=0.1):
    # Find frequent itemsets using FP-Growth
    frequent_itemsets = fpgrowth(basket_sets, min_support=min_support, use_colnames=True)
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by='confidence', ascending=False)
    return rules

# Function to recommend products based on a given customer's basket
def recommend_products(rules, basket_items):
    # This function expects basket_items as a list of products
    antecedents = set(basket_items)
    recommendations = rules[rules['antecedents'] == antecedents][['consequents', 'confidence']]
    return recommendations

# Main function to run the Streamlit app
def main():
    st.title("E-commerce Product Recommendation System")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        basket_sets = load_data(uploaded_file)
        st.success("Data successfully loaded and preprocessed.")
        
        # Parameters for FP-Growth
        min_support = st.slider("Minimum support:", 0.01, 0.1, 0.01)
        min_confidence = st.slider("Minimum confidence:", 0.1, 1.0, 0.1)
        
        rules = apply_fp_growth(basket_sets, min_support, min_confidence)
        st.success("FP-Growth analysis completed. Rules generated.")
        
        # Input for current basket
        user_basket_input = st.text_input("Enter a list of products in the basket (comma separated):")
        if user_basket_input:
            basket_items = user_basket_input.split(',')
            recommendations = recommend_products(rules, basket_items)
            if not recommendations.empty:
                st.write("Recommendations based on the current basket:")
                st.write(recommendations)
            else:
                st.write("No recommendations found for the current basket.")

if __name__ == '__main__':
    main()
