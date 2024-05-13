# Import necessary libraries
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Function to load and preprocess data
def load_data(file_path):
    # Load the transaction data
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data['GroupPrice'] = data['Quantity'] * data['UnitPrice']
    data = data.dropna()
    # Remove gift items
    data = data[data['StockCode'].str.contains('^[1-9]')]
    return data

# Prepare data for the FP-growth algorithm
def prepare_basket(data):
    basket = data.groupby(['InvoiceNo', 'CustomerID'])['StockCode'].apply(set).reset_index()
    return basket

# Run FP-growth algorithm
def run_fpgrowth(basket, min_support=0.01, min_confidence=0.1):
    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules.sort_values('confidence', ascending=False)
    return rules

# Main function to run the Streamlit app
def main():
    st.title("E-commerce Product Recommendation System")
    
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        basket = prepare_basket(data)
        st.success("Data successfully loaded and preprocessed.")
        
        min_support = st.slider("Minimum support", 0.01, 0.1, 0.01)
        min_confidence = st.slider("Minimum confidence", 0.1, 1.0, 0.1)
        
        rules = run_fpgrowth(basket, min_support, min_confidence)
        st.success("FP-Growth analysis completed. Rules generated.")
        
        # Display rules
        st.write(rules.head())

if __name__ == '__main__':
    main()

