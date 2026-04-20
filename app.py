import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

# Set page config
st.set_page_config(
    page_title="SmartCart - Customer Groups",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #E8F4F8;
        border-left: 5px solid #2E86AB;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-text {
        font-size: 1rem;
        color: #333;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CACHING & FUNCTIONS
# ============================================

@st.cache_data
def load_data():
    """Load and preprocess data"""
    df = pd.read_csv("smartcart_customers.csv")
    return df

@st.cache_data
def preprocess_data(df):
    """Data preprocessing pipeline"""
    df_copy = df.copy()
    
    # Handle missing values
    df_copy["Income"] = df_copy["Income"].fillna(df_copy["Income"].median())
    
    # Feature engineering
    df_copy["Age"] = 2026 - df_copy["Year_Birth"]
    df_copy["Dt_Customer"] = pd.to_datetime(df_copy["Dt_Customer"], dayfirst=True)
    reference_date = df_copy["Dt_Customer"].max()
    df_copy["Customer_Tenure_Days"] = (reference_date - df_copy["Dt_Customer"]).dt.days
    df_copy["Total_Spending"] = (df_copy["MntWines"] + df_copy["MntFruits"] + 
                                  df_copy["MntMeatProducts"] + df_copy["MntFishProducts"] + 
                                  df_copy["MntSweetProducts"] + df_copy["MntGoldProds"])
    df_copy["Total_Children"] = df_copy["Kidhome"] + df_copy["Teenhome"]
    
    # Education mapping
    df_copy["Education"] = df_copy["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    
    # Marital status mapping
    df_copy["Living_With"] = df_copy["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    })
    
    # Remove outliers
    df_copy = df_copy[(df_copy["Age"] < 90) & (df_copy["Income"] < 600_000)]
    
    # Drop unnecessary columns (but keep Total_Spending and Total_Children)
    cols_to_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer",
                    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df_copy = df_copy.drop(columns=cols_to_drop)
    
    # One-hot encoding
    ohe = OneHotEncoder()
    cat_cols = ["Education", "Living_With"]
    enc_cols = ohe.fit_transform(df_copy[cat_cols])
    enc_df = pd.DataFrame(enc_cols.toarray(), columns=ohe.get_feature_names_out(cat_cols), index=df_copy.index)
    
    # Keep numeric columns and add encoded categorical columns
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    df_final = pd.concat([df_copy[numeric_cols], enc_df], axis=1)
    
    return df_final

@st.cache_data
def scale_and_pca(df):
    """Scale data and apply PCA"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_scaled, X_pca, pca, scaler

@st.cache_data
def elbow_method(X_pca):
    """Calculate WCSS for elbow method"""
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)
    
    knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
    return wcss, knee.elbow

@st.cache_data
def silhouette_analysis(X_pca):
    """Calculate silhouette scores"""
    scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        scores.append(score)
    return scores

@st.cache_data
def perform_clustering(X_pca, n_clusters=4, method="hierarchical"):
    """Perform clustering"""
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_pca)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_pca)
    
    return labels, model

# ============================================
# MAIN APP
# ============================================

# Header
st.markdown("<h1 class='main-header'>🛒 SmartCart - Find Your Customer Groups</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understand your customers and find similar groups to target better</p>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
💡 <b>What is this?</b> This tool analyzes your customer data and automatically finds groups of similar customers 
(like "Big Spenders", "Budget Shoppers", "Loyal Customers", etc.) so you can target them better!
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🎯 Settings")
    st.write("Adjust these settings to find the best customer groups:")
    
    n_clusters = st.slider(
        "How many customer groups do you want?", 
        min_value=2, 
        max_value=10, 
        value=4,
        help="More groups = more specific targeting. Fewer groups = simpler overview."
    )
    
    st.markdown("---")
    st.write("**Finding Method** (Advanced):")
    clustering_method = st.radio(
        "Choose a method", 
        ["Simple Method", "Advanced Method"],
        help="Simple = Faster, Advanced = Sometimes more accurate"
    )
    method_map = {"Simple Method": "kmeans", "Advanced Method": "hierarchical"}
    method = method_map[clustering_method]

# Load and preprocess data
try:
    df_raw = load_data()
    df_processed = preprocess_data(df_raw)
    X_scaled, X_pca, pca, scaler = scale_and_pca(df_processed)
    
    st.success("✅ Data loaded and preprocessed successfully!")
    
    # Tab 1: Data Overview
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Your Customers", 
        "📊 Quick Stats", 
        "🔎 Finding Groups", 
        "🎯 Your Groups", 
        "💼 Group Details"
    ])
    
    with tab1:
        st.subheader("👥 Your Customer Base")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Customers", len(df_raw))
        with col2:
            st.metric("✅ Cleaned Data", len(df_processed))
        with col3:
            st.metric("❌ Removed Outliers", len(df_raw) - len(df_processed))
        with col4:
            st.metric("📋 Info Tracked", df_processed.shape[1])
        
        st.markdown("---")
        st.write("#### Sample Customers (First 10)")
        st.info("This shows real customer data from your system")
        display_cols = ['Year_Birth', 'Education', 'Income', 'Recency', 'NumWebPurchases', 'NumStorePurchases']
        available_cols = [col for col in display_cols if col in df_raw.columns]
        st.dataframe(df_raw[available_cols].head(10), use_container_width=True)
        
        st.write("#### Summary of Your Data")
        st.info("Average values across all customers")
        summary_data = {
            'Metric': ['Average Age', 'Average Income', 'Avg Children', 'Avg Days Since Last Purchase'],
            'Value': [
                f"{df_processed['Age'].mean():.1f} years",
                f"${df_processed['Income'].mean():,.0f}",
                f"{df_processed['Total_Children'].mean():.1f}",
                f"{df_processed['Recency'].mean():.0f} days"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("📊 Quick Look at Your Customers")
        
        st.markdown("""
        <div class='info-box'>
        These charts show patterns in your customer base - who spends the most, 
        age distribution, and which factors matter most.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### How Old Are Your Customers?")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df_processed["Age"], bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Age Distribution")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.write("#### What Are Their Income Levels?")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df_processed["Income"], bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
            ax.set_xlabel("Income ($)")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Income Distribution")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        st.write("#### What Factors Influence Spending?")
        st.info("This shows which customer traits have the strongest relationship with how much they spend")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df_processed.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, cbar_kws={"shrink": 0.8}, center=0)
        plt.title("How Customer Traits Connect (Darker Red = Strong Connection)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.write("#### Money vs Spending Pattern")
        st.info("Do richer customers spend more? This chart shows the relationship")
        fig = px.scatter(df_processed, x='Income', y='Total_Spending', 
                        title='Income vs How Much Customers Spend',
                        labels={'Income': 'Income ($)', 'Total_Spending': 'Total Spending ($)'},
                        opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔎 Finding the Best Number of Groups")
        
        st.markdown("""
        <div class='info-box'>
        <b>How do we find the right number of groups?</b><br>
        The system tests different group sizes and scores them. <br>
        Think of it like: Too few groups = oversimplified. Too many = too complex.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Method 1: Find the 'Sweet Spot' 📍")
            st.write("Shows where adding more groups stops helping much")
            wcss, optimal_k_elbow = elbow_method(X_pca)
            
            fig = px.line(x=list(range(1, 11)), y=wcss, markers='o',
                         title=f'Finding Sweet Spot (Suggested: {optimal_k_elbow} groups)',
                         labels={'x': 'Number of Groups', 'y': 'Compactness Score'})
            fig.add_annotation(
                x=optimal_k_elbow, y=wcss[optimal_k_elbow-1],
                text=f"Sweet Spot: {optimal_k_elbow}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✨ **Recommended:** {optimal_k_elbow} groups would work well!")
        
        with col2:
            st.write("#### Method 2: Quality Score 🎯")
            st.write("Higher score = Better quality groups (more separated)")
            scores = silhouette_analysis(X_pca)
            best_k_silhouette = scores.index(max(scores)) + 2
            
            fig = px.line(x=list(range(2, 11)), y=scores, markers='o',
                         title=f'Group Quality Score (Best: {best_k_silhouette} groups)',
                         labels={'x': 'Number of Groups', 'y': 'Quality Score (0-1)'})
            fig.add_annotation(
                x=best_k_silhouette, y=max(scores),
                text=f"Best Score: {best_k_silhouette}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="green"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✨ **Best Quality:** {best_k_silhouette} groups!")
    
    with tab4:
        st.subheader("🎯 Your Customer Groups")
        
        # Perform clustering
        labels, model = perform_clustering(X_pca, n_clusters, method)
        df_processed["Cluster"] = labels
        
        st.markdown("""
        <div class='info-box'>
        The system found <b>""" + str(n_clusters) + """ different groups of similar customers</b>. 
        Each group has unique characteristics you can target separately!
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### How Many Customers in Each Group?")
            cluster_counts = df_processed["Cluster"].value_counts().sort_index()
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        title='Customers Per Group',
                        labels={'x': 'Group Number', 'y': 'Number of Customers'},
                        color=cluster_counts.index,
                        text=cluster_counts.values)
            fig.update_traces(textposition='auto')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("#### Where Are Your Groups? (Visual Map)")
            st.info("This shows your groups spread across different customer dimensions")
            fig = px.scatter_3d(
                x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                color=labels,
                title='All Customer Groups (3D View)',
                labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("#### Money vs Spending by Group")
        st.info("Which groups spend the most? Which earn the most?")
        fig = px.scatter(df_processed, x='Total_Spending', y='Income',
                        color='Cluster',
                        size='Total_Children',
                        title='Income vs Spending (Size = Family Size)',
                        labels={'Total_Spending': 'Total Spending ($)', 'Income': 'Income ($)', 'Cluster': 'Group'},
                        color_continuous_scale='Viridis',
                        opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("💼 Meet Your Customer Groups")
        
        st.markdown("""
        <div class='info-box'>
        Below is a detailed breakdown of each group. Learn who they are, 
        what they like, and how to target them!
        </div>
        """, unsafe_allow_html=True)
        
        # Individual cluster insights
        for cluster_id in sorted(df_processed["Cluster"].unique()):
            cluster_data = df_processed[df_processed["Cluster"] == cluster_id]
            
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"### Group {cluster_id}")
            
            with col3:
                st.metric("👥 Size", f"{len(cluster_data)} customers")
            
            # Calculate characteristics
            avg_spending = cluster_data['Total_Spending'].mean()
            avg_income = cluster_data['Income'].mean()
            avg_age = cluster_data['Age'].mean()
            children_count = cluster_data['Total_Children'].mean()
            conversion_rate = cluster_data['Response'].mean() * 100
            avg_recency = cluster_data['Recency'].mean()
            web_purchases = cluster_data['NumWebPurchases'].mean()
            
            # Create persona
            if avg_spending > df_processed['Total_Spending'].quantile(0.75) and avg_income > df_processed['Income'].quantile(0.75):
                persona = "💎 VIP Customers"
                persona_desc = "High earners who spend generously. These are your best customers!"
                color = "#FFD700"
            elif avg_income < df_processed['Income'].quantile(0.25):
                persona = "💰 Budget Shoppers"
                persona_desc = "Lower income but loyal. Great for promoting affordable items."
                color = "#90EE90"
            elif children_count > df_processed['Total_Children'].quantile(0.75):
                persona = "👨‍👩‍👧‍👦 Family Focused"
                persona_desc = "Families with kids. Target family bundles and kid products."
                color = "#FF69B4"
            elif avg_age < df_processed['Age'].quantile(0.25):
                persona = "🎯 Young & Active"
                persona_desc = "Younger crowd, tech-savvy. Use online channels."
                color = "#1E90FF"
            elif avg_recency < df_processed['Recency'].quantile(0.25):
                persona = "⭐ Highly Active"
                persona_desc = "Frequent buyers. Great engagement potential!"
                color = "#FF6347"
            else:
                persona = "⭐ Regular Customers"
                persona_desc = "Steady, reliable customers. Good base for campaigns."
                color = "#FFB6C1"
            
            # Display with color box
            st.markdown(f"""
            <div style='background-color: {color}; padding: 15px; border-radius: 8px; color: white; margin: 10px 0;'>
                <h4>{persona}</h4>
                <p>{persona_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key stats in columns
            stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
            
            with stat_col1:
                st.metric("📅 Avg Age", f"{avg_age:.0f} yrs")
            with stat_col2:
                st.metric("💵 Avg Income", f"${avg_income:,.0f}")
            with stat_col3:
                st.metric("🛍️ Total Spent", f"${avg_spending:,.0f}")
            with stat_col4:
                st.metric("👶 Avg Kids", f"{children_count:.1f}")
            with stat_col5:
                st.metric("🎁 Conversion", f"{conversion_rate:.1f}%")
            
            # Business insights
            st.write("**📊 What You Should Know About This Group:**")
            
            insights = []
            
            if avg_spending > df_processed['Total_Spending'].mean() * 1.5:
                insights.append("✅ Big spenders - Premium products work well")
            if avg_spending < df_processed['Total_Spending'].mean() * 0.7:
                insights.append("✅ Price-sensitive - Offer discounts & deals")
            if children_count > 1:
                insights.append("✅ Family households - Bundle deals recommended")
            if conversion_rate > 50:
                insights.append("✅ Highly responsive - They open marketing emails!")
            if conversion_rate < 10:
                insights.append("✅ Low engagement - Try re-engagement campaigns")
            if web_purchases > cluster_data['NumStorePurchases'].mean():
                insights.append("✅ Online shoppers - Focus on web marketing")
            else:
                insights.append("✅ In-store preference - Promote store visits")
            if avg_age < 40:
                insights.append("✅ Younger group - Use social media & mobile")
            else:
                insights.append("✅ Older group - Traditional channels work better")
            
            for insight in insights:
                st.write(f"  {insight}")
            
            st.markdown("---")
        
        st.write("#### 📈 Compare Groups Side-by-Side")
        metrics_to_plot = st.multiselect(
            "Which metrics would you like to compare?",
            ["Age", "Income", "Total_Spending", "Total_Children", "Customer_Tenure_Days", "Response"],
            default=["Age", "Income", "Total_Spending"]
        )
        
        if metrics_to_plot:
            for metric in metrics_to_plot:
                metric_label = metric.replace('_', ' ')
                fig = px.box(df_processed, x='Cluster', y=metric,
                            title=f'How does "{metric_label}" vary across groups?',
                            color='Cluster',
                            points="all")
                st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Make sure `smartcart_customers.csv` is in the same directory as this script.")
