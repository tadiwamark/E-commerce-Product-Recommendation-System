# Import necessary libraries
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import plotly.express as px

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(github_url, encoding='ISO-8859-1')
    data = data.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'UnitPrice'])
    
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

# Function to recommend a product based on the current basket
def recommend_product(basket, rules, data):
    recommendations = []
    for items in basket:
        applicable_rules = rules[rules['antecedents'] == frozenset(items)]
        for _, rule in applicable_rules.iterrows():
            consequents = rule['consequents']
            for consequent in consequents:
                item_details = data[data['StockCode'] == consequent]
                if not item_details.empty:
                    description = item_details['Description'].values[0]
                    price = item_details['UnitPrice'].values[0]
                    recommendations.append((consequent, description, rule['confidence'], price))
    if recommendations:
        return recommendations
    else:
        return "No recommendation available"

# Main function to run the Streamlit app
def main():
    st.title("E-commerce Product Recommendation System")
    
    github_url = 'https://github.com/tadiwamark/E-commerce-Product-Recommendation-System/blob/main/data.csv'
    if github_url:
        data = load_data(github_url )
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


        product_input = st.text_input("Enter products in the basket (comma-separated):")
        
        if product_input:
            user_basket = set(product_input.split(','))
            recommendation = recommend_product([user_basket], rules, data)
            st.write("Recommended Products:")
            for item in recommendation:
                st.write(f"Product Code: {item[0]}, Description: {item[1]}, Probability: {item[2]:.2f}, Estimated Price: {item[3]*item[2]:.2f}")




        if st.button('Download Rules as CSV'):
            st.download_button('Download', data=rules.to_csv(index=False), file_name='association_rules.csv', mime='text/csv')







if __name__ == '__main__':
    main()

