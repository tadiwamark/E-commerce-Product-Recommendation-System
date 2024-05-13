# Import necessary libraries
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import plotly.express as px

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data = data.dropna(subset=['InvoiceNo', 'StockCode'])
    
    return data

# Prepare data for the FP-growth algorithm
def prepare_basket(data):
    # Grouping products by InvoiceNo and CustomerID
    basket = (data.groupby(['InvoiceNo', 'StockCode'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .drop('InvoiceNo', axis=1))
    # One-hot encoding
    basket = (basket > 0).astype(int)
    return basket

# Run FP-growth algorithm
def run_fpgrowth(basket, min_support=0.01, min_confidence=0.1):
    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules.sort_values('confidence', ascending=False)
    return rules



def visualize_rules(rules):
    fig = px.scatter(rules.head(20), x='support', y='confidence', hover_data=['antecedents', 'consequents'], size='lift', color='lift')
    st.plotly_chart(fig, use_container_width=True)


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
        
        product_filter = st.multiselect('Filter rules by product:', options=data['StockCode'].unique())
        
        if product_filter:
            filtered_rules = rules[rules['antecedents'].apply(lambda x: any(item in x for item in product_filter)) |
                            rules['consequents'].apply(lambda x: any(item in x for item in product_filter))]
            st.write(filtered_rules)


        user_input = st.text_input("Enter a product or set of products (comma-separated):")
        
        if user_input:
            input_set = set(user_input.split(','))
            relevant_rules = rules[rules['antecedents'].apply(lambda x: x == input_set)]
            st.write(relevant_rules)


        if st.button('Download Rules as CSV'):
            st.download_button('Download', data=rules.to_csv(index=False), file_name='association_rules.csv', mime='text/csv')







if __name__ == '__main__':
    main()

