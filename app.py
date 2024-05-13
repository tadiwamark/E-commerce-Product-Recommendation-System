# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Function to load and preprocess data
def load_data():
    data = pd.read_csv('data.csv', encoding='ISO-8859-1')  
    # Convert transactions to a one-hot encoded dataframe
    data = pd.get_dummies(data, columns=['Product'], prefix='', prefix_sep='')
    return data


# Function to apply FP-Growth and find frequent item sets
def apply_fp_growth(data):
    frequent_itemsets = fpgrowth(data, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    rules = rules.sort_values(by='confidence', ascending=False)
    return rules

# Function to recommend products based on a given basket
def recommend_products(rules, basket):
    # Convert basket to set and find recommendations
    basket_set = frozenset(basket.split(','))
    recommendations = rules[rules['antecedents'] == basket_set]['consequents']
    return recommendations

# Main function to run the Streamlit app
def main():
    st.title("E-commerce Product Recommendation System")

    data = load_data()
    st.write("Data successfully loaded and preprocessed.")
        
    rules = apply_fp_growth(data)
    st.write("FP-Growth analysis completed.")

    # User input for current basket
    customer_basket = st.text_input("Enter the current basket items separated by comma")
    if customer_basket:
        recommendations = recommend_products(rules, customer_basket)
        st.write("Recommended Products:")
        for index, row in recommendations.iterrows():
            st.write(f"{list(row['consequents'])} with confidence {row['confidence']:.2f}")

if __name__ == '__main__':
    main()
