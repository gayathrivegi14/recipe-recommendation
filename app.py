import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast
from collections import Counter
import warnings
import json # To parse nutrition column if it's stored as a stringified JSON

warnings.filterwarnings('ignore')

# METRICS FUNCTIONS
def is_match(item, relevant_list, threshold=60):
    item = item.lower()
    return any(rel.lower() in item for rel in relevant_list)

def precision_at_k(recommended, relevant, k=5):
    return sum(1 for item in recommended[:k] if is_match(item, relevant)) / k if k else 0

def recall_at_k(recommended, relevant, k=5):
    matched = set()
    for rel in relevant:
        for item in recommended[:k]:
            if is_match(item, [rel]):
                matched.add(rel)
                break
    return len(matched) / len(relevant) if relevant else 0

def f1_score(p, r):
    return 0 if (p + r) == 0 else 2 * (p * r) / (p + r)

def mmr_ranking(similarity_scores, lambda_param=0.7, top_k=5):
    selected, candidates = [], list(range(len(similarity_scores)))
    for _ in range(min(top_k, len(similarity_scores))):
        scores = [(i, lambda_param * similarity_scores[i] -
                   (1 - lambda_param) * max([similarity_scores[j] for j in selected], default=0))
                  for i in candidates]
        best = max(scores, key=lambda x: x[1])[0]
        selected.append(best)
        candidates.remove(best)
    return selected

# --- Helper Functions (Copied and adapted from your base code) ---

def load_data(file_path):
    """
    Load your recipes.csv dataset
    """
    try:
        df = pd.read_csv(file_path)
        # st.success(f"✅ Dataset loaded successfully with {len(df)} recipes")
        # st.sidebar.write(f"📊 Columns available: {list(df.columns)}")
        return clean_dataframe(df)
    except FileNotFoundError:
        st.error(f"❌ Error: File '{file_path}' not found.")
        st.info("💡 Please make sure 'recipes.csv' is in the same directory as this app")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

def clean_dataframe(df):
    """
    Clean the dataframe and convert columns to proper data types
    """
    df_clean = df.copy()

    # Convert numeric columns, handling errors and missing values
    numeric_columns = ['prep_time', 'cook_time', 'total_time', 'servings', 'rating']

    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(extract_number)

    # Handle missing values for text columns
    text_columns = ['ingredients', 'directions', 'recipe_name', 'url', 'cuisine_path', 'nutrition', 'timing', 'img_src']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('')

    # Fill NaN numeric values with 0
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    # Convert 'nutrition' column from string to dictionary if it exists
    if 'nutrition' in df_clean.columns:
        df_clean['nutrition'] = df_clean['nutrition'].apply(parse_nutrition)

    return df_clean

def extract_number(value):
    """
    Extract numeric value from mixed data types
    """
    if pd.isna(value) or value == '':
        return 0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        numbers = re.findall(r'\d+\.?\d*', value)
        if numbers:
            return float(numbers[0])
        time_mapping = {
            'hour': 60, 'hours': 60, 'hr': 60, 'hrs': 60,
            'minute': 1, 'minutes': 1, 'min': 1, 'mins': 1
        }
        for unit, multiplier in time_mapping.items():
            if unit in value.lower():
                numbers = re.findall(r'\d+\.?\d*', value)
                if numbers:
                    return float(numbers[0]) * multiplier
    return 0

def parse_nutrition(nutrition_str):
    """
    Parses nutrition string from CSV format like "Total Fat 18g 23%, Saturated Fat 7g 34%..."
    into a structured dictionary with calories, fat, carbs, protein
    """
    if pd.isna(nutrition_str) or nutrition_str == '':
        return {}
    
    nutrition_dict = {}
    
    try:
        # If it's already a dict, return it
        if isinstance(nutrition_str, dict):
            return nutrition_str
            
        nutrition_str = str(nutrition_str)
        
        # Extract calories (format: "Calories 250" or similar)
        calories_match = re.search(r'(\d+)\s*calories', nutrition_str, re.IGNORECASE)
        if calories_match:
            nutrition_dict['calories'] = float(calories_match.group(1))
        
        # Extract total fat (format: "Total Fat 18g")
        fat_match = re.search(r'Total Fat\s*(\d+\.?\d*)g', nutrition_str, re.IGNORECASE)
        if fat_match:
            nutrition_dict['fatContent'] = float(fat_match.group(1))
        
        # Extract carbohydrates (format: "Total Carbohydrate 60g")
        carb_match = re.search(r'Total Carbohydrate\s*(\d+\.?\d*)g', nutrition_str, re.IGNORECASE)
        if carb_match:
            nutrition_dict['carbohydrateContent'] = float(carb_match.group(1))
        
        # Extract protein (format: "Protein 4g")
        protein_match = re.search(r'Protein\s*(\d+\.?\d*)g', nutrition_str, re.IGNORECASE)
        if protein_match:
            nutrition_dict['proteinContent'] = float(protein_match.group(1))
        
        return nutrition_dict
        
    except Exception as e:
        return {}


class RecipeRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.preprocess_data()
        # Limiting max_features to keep the vectorizer size manageable for Streamlit deployment
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        self.ingredient_matrix = self._create_ingredient_matrix()

    def preprocess_data(self):
        self.df['cleaned_ingredients'] = self.df['ingredients'].apply(self._clean_ingredients)

    def _clean_ingredients(self, ingredients_text):
        if pd.isna(ingredients_text) or ingredients_text == '':
            return ""

        ingredients_str = str(ingredients_text)

        try:
            # Split by comma - ingredients are comma-separated in the CSV
            ingredients_list = re.split(r',', ingredients_str)

            cleaned_ingredients = []
            for ingredient in ingredients_list:
                if isinstance(ingredient, str):
                    # Remove measurements and numbers
                    cleaned = re.sub(r'\d+/\d+|\d+\.\d+|\d+', '', ingredient)
                    # Remove special characters but keep spaces
                    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
                    # Remove extra whitespace and convert to lowercase
                    cleaned = ' '.join(cleaned.split()).strip().lower()
                    # Filter out very short words and common stop words
                    if cleaned and len(cleaned) > 2:
                        cleaned_ingredients.append(cleaned)

            return ' '.join(cleaned_ingredients)

        except Exception:
            cleaned = re.sub(r'\d+/\d+|\d+\.\d+|\d+', '', ingredients_str)
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            return ' '.join(cleaned.split()).lower()

    def _create_ingredient_matrix(self):
        ingredient_corpus = self.df['cleaned_ingredients'].tolist()
        matrix = self.vectorizer.fit_transform(ingredient_corpus)
        return matrix

    def find_recipes_by_ingredients(self, user_ingredients, top_n=10, min_similarity=0.1,
                                    max_calories=None, max_fat=None, max_carbs=None, max_protein=None,
                                    dietary_prefs=None):
        user_ingredients_clean = self._clean_ingredients(user_ingredients)

        if not user_ingredients_clean:
            # If no ingredients, apply only nutrition filters if present
            filtered_df = self.df.copy()
            st.warning("⚠️ No valid ingredients detected. Applying nutrition filters only.")
            return self._apply_nutrition_filters(filtered_df, max_calories, max_fat, max_carbs, max_protein, dietary_prefs).head(top_n)


        user_vector = self.vectorizer.transform([user_ingredients_clean])
        similarities = cosine_similarity(user_vector, self.ingredient_matrix).flatten()

        similar_indices = similarities.argsort()[::-1]
        potential_matches_df = self.df.iloc[similar_indices].copy()
        potential_matches_df['similarity_score'] = similarities[similar_indices]

        # Filter by minimum similarity first
        filtered_by_similarity = potential_matches_df[potential_matches_df['similarity_score'] >= min_similarity]

        if filtered_by_similarity.empty:
            st.warning("❌ No recipes found matching your ingredients. Try different ingredients.")
            return pd.DataFrame()

        # Apply nutrition filters
        final_recommendations = self._apply_nutrition_filters(
            filtered_by_similarity, max_calories, max_fat, max_carbs, max_protein, dietary_prefs
        )

        return self._format_results(final_recommendations).head(top_n)

    def _apply_nutrition_filters(self, df_to_filter, max_calories, max_fat, max_carbs, max_protein, dietary_prefs):
        filtered_df = df_to_filter.copy()

        # Nutrition filters - only apply if values are set and nutrition data exists
        if max_calories is not None and max_calories > 0:
            filtered_df = filtered_df[filtered_df['nutrition'].apply(
                lambda x: x.get('calories', float('inf')) <= max_calories if isinstance(x, dict) and x.get('calories') else True
            )]
        if max_fat is not None and max_fat > 0:
            filtered_df = filtered_df[filtered_df['nutrition'].apply(
                lambda x: x.get('fatContent', float('inf')) <= max_fat if isinstance(x, dict) and x.get('fatContent') else True
            )]
        if max_carbs is not None and max_carbs > 0:
            filtered_df = filtered_df[filtered_df['nutrition'].apply(
                lambda x: x.get('carbohydrateContent', float('inf')) <= max_carbs if isinstance(x, dict) and x.get('carbohydrateContent') else True
            )]
        if max_protein is not None and max_protein > 0:
            filtered_df = filtered_df[filtered_df['nutrition'].apply(
                lambda x: x.get('proteinContent', 0) >= max_protein if isinstance(x, dict) and x.get('proteinContent') else False
            )]

        # Dietary preference filters
        if dietary_prefs:
            for pref in dietary_prefs:
                if pref.lower() == 'vegetarian':
                    filtered_df = filtered_df[
                        ~filtered_df['cleaned_ingredients'].str.contains('chicken|beef|pork|fish|lamb|bacon|meat', regex=True, na=False)
                    ]
                elif pref.lower() == 'vegan':
                    filtered_df = filtered_df[
                        ~filtered_df['cleaned_ingredients'].str.contains('chicken|beef|pork|fish|lamb|bacon|meat|dairy|milk|cheese|egg|butter|cream|yogurt|honey', regex=True, na=False)
                    ]
                elif pref.lower() == 'gluten-free':
                    filtered_df = filtered_df[
                        ~filtered_df['cleaned_ingredients'].str.contains('wheat|flour|bread|pasta|barley|rye|gluten', regex=True, na=False)
                    ]
        
        # Sort by similarity score if it exists
        if 'similarity_score' in filtered_df.columns:
            return filtered_df.sort_values(by='similarity_score', ascending=False)
        return filtered_df

    def _format_results(self, results):
        required_columns = [
            'recipe_name', 'prep_time', 'cook_time', 'total_time',
            'servings', 'ingredients', 'directions', 'rating', 'url',
            'cuisine_path', 'nutrition', 'timing', 'img_src', 'similarity_score'
        ]

        for col in required_columns:
            if col not in results.columns:
                results[col] = ''

        return results[required_columns]


def advanced_analysis(df):
    """
    Perform advanced analysis on the recipe dataset for Streamlit display
    """
    st.subheader("📊 Database Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Recipes", f"{len(df):,}")
    col2.metric("Average Rating", f"{df['rating'].mean():.2f}/5")
    col3.metric("Average Total Time", f"{df['total_time'].mean():.1f} min")

    st.markdown("---")
    st.write("📈 Rating Distribution:")
    rating_counts = df['rating'].value_counts().sort_index(ascending=False)
    st.bar_chart(rating_counts)

    if 'cuisine_path' in df.columns:
        st.write("🍽️ Top Cuisine Categories:")
        cuisine_counts = df['cuisine_path'].value_counts().head(8)
        st.bar_chart(cuisine_counts)

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Healthy Recipe Recommender",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍳 Healthy Recipe Recommender")
st.write("Find delicious recipes based on what you have at home and your nutrition goals!")

# Load data only once
@st.cache_resource
def get_recommender():
    df = load_data('recipes.csv')
    if df is not None:
        recommender = RecipeRecommender(df)
        return recommender, df
    return None, None

recommender, df = get_recommender()

if recommender is None:
    st.stop() # Stop if data loading failed

# Sidebar for inputs
with st.sidebar:
    st.header("Your Ingredients & Goals")
    user_ingredients = st.text_area(
        "What ingredients do you have?",
        "chicken, rice, broccoli",
        height=100,
        help="Enter ingredients separated by commas (e.g., chicken, rice, broccoli)"
    )
    top_n_recipes = st.slider("Number of recommendations", 3, 20, 10)
    min_similarity_threshold = st.slider("Ingredient Match Sensitivity", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")
    st.subheader("Nutrition Goals (Optional)")
    enable_nutrition = st.checkbox("Filter by nutrition goals")
    
    if enable_nutrition:
        max_calories = st.number_input("Max Calories", min_value=0, max_value=5000, value=500, step=50)
        max_fat = st.number_input("Max Fat (g)", min_value=0, max_value=200, value=30, step=5)
        max_carbs = st.number_input("Max Carbs (g)", min_value=0, max_value=500, value=70, step=10)
        min_protein = st.number_input("Min Protein (g)", min_value=0, max_value=300, value=15, step=5)
    else:
        max_calories = None
        max_fat = None
        max_carbs = None
        min_protein = None

    st.markdown("---")
    st.subheader("Dietary Preferences")
    dietary_prefs = st.multiselect(
        "Select all that apply:",
        ['Vegetarian', 'Vegan', 'Gluten-Free']
    )

if st.button("Find Recipes!", type="primary"):
    st.subheader(f"🍽️ Your Recommended Recipes")
    if not user_ingredients.strip():
        st.warning("Please enter at least one ingredient to get recommendations.")
    else:
        with st.spinner("Searching for delicious and healthy recipes..."):
            recommendations = recommender.find_recipes_by_ingredients(
                user_ingredients=user_ingredients,
                top_n=top_n_recipes,
                min_similarity=min_similarity_threshold,
                max_calories=max_calories,
                max_fat=max_fat,
                max_carbs=max_carbs,
                max_protein=min_protein, # Pass min_protein here
                dietary_prefs=dietary_prefs
            )

            if recommendations.empty:
                st.info("No recipes found matching your criteria. Try adjusting your ingredients or nutrition goals.")
            else:
                for idx, recipe in recommendations.iterrows():
                    st.markdown("---")
                    col1_rec, col2_rec = st.columns([1, 3])
                    with col1_rec:
                        if recipe['img_src']:
                            st.image(recipe['img_src'], width=200)
                        else:
                            st.image("https://via.placeholder.com/200x200?text=No+Image", width=200)
                            
                    with col2_rec:
                        st.subheader(f"{recipe['recipe_name']}")
                        st.write(f"⭐ Rating: {recipe['rating']}/5")
                        st.write(f"📊 Match Score: {recipe['similarity_score']:.2f}")
                        st.write(f"⏱️ Total Time: {recipe['total_time']} min | Servings: {recipe['servings']}")
                        if recipe['cuisine_path']:
                            st.write(f"🍽️ Cuisine: {recipe['cuisine_path']}")

                        # Display key nutrition facts
                        nutrition_info = recipe['nutrition']
                        if isinstance(nutrition_info, dict) and nutrition_info:
                            cal = nutrition_info.get('calories', 'N/A')
                            fat = nutrition_info.get('fatContent', 'N/A')
                            carbs = nutrition_info.get('carbohydrateContent', 'N/A')
                            protein = nutrition_info.get('proteinContent', 'N/A')
                            
                            if cal != 'N/A' or fat != 'N/A' or carbs != 'N/A' or protein != 'N/A':
                                st.markdown(f"**Nutrition (per serving):** "
                                            f"Cal: {cal} | "
                                            f"Fat: {fat}g | "
                                            f"Carbs: {carbs}g | "
                                            f"Protein: {protein}g")
                            else:
                                st.info("Nutrition info not available")
                        else:
                            st.info("Nutrition info not available")

                        with st.expander("View Full Details"):
                            st.markdown(f"**Ingredients:**")
                            ingredients = recipe['ingredients']
                            if pd.notna(ingredients) and ingredients != '':
                                # Split by comma since ingredients are comma-separated
                                ingredients_list = [ing.strip() for ing in str(ingredients).split(',')]
                                for ingredient in ingredients_list:
                                    if ingredient:
                                        st.write(f"- {ingredient}")
                            else:
                                st.write("No ingredients listed")

                            st.markdown(f"**Directions:**")
                            directions = recipe['directions']
                            if pd.notna(directions) and directions != '':
                                # Split by newlines or periods for better formatting
                                directions_text = str(directions)
                                # Try to split into steps
                                steps = re.split(r'\n+', directions_text)
                                if len(steps) > 1:
                                    for i, step in enumerate(steps, 1):
                                        if step.strip():
                                            st.write(f"{i}. {step.strip()}")
                                else:
                                    st.write(directions_text)
                            else:
                                st.write("No directions available")

                            if recipe['url']:
                                st.markdown(f"[🔗 View Original Recipe]({recipe['url']})")
                            
# Optional: Display advanced analysis in a separate tab or section
st.markdown("---")
st.header("Database Insights")
advanced_analysis(df)

# MODEL EVALUATION 
st.markdown("---")
st.header("📊 Model Evaluation")

if st.button("Run Evaluation"):

    precision_list, recall_list, f1_list = [], [], []

    test_data = []
    for i in range(min(10, len(df))):
        name = df.iloc[i]['recipe_name']
        ingredients = df.iloc[i]['ingredients']
        query = str(ingredients)
        test_data.append({"query": query, "relevant": [name]})

    for test in test_data:
        results = recommender.find_recipes_by_ingredients(test["query"])

        if results.empty:
            continue

        recommended = results['recipe_name'].tolist()
        scores = results['similarity_score'].values

        mmr_idx = mmr_ranking(scores)
        mmr_rec = [recommended[i] for i in mmr_idx if i < len(recommended)]

        p = precision_at_k(mmr_rec, test["relevant"])
        r = recall_at_k(mmr_rec, test["relevant"])
        f1 = f1_score(p, r)

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)

    if precision_list:
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", round(np.mean(precision_list), 3))
        col2.metric("Recall", round(np.mean(recall_list), 3))
        col3.metric("F1 Score", round(np.mean(f1_list), 3))

        st.subheader("📈 Performance Graph")

        # Create DataFrame for plotting
        metrics_df = pd.DataFrame({
            "Precision": precision_list,
            "Recall": recall_list,
            "F1 Score": f1_list
        })

        # Show line chart
        st.line_chart(metrics_df)
        
    else:
        st.warning("No evaluation results")


st.sidebar.markdown("---")
st.sidebar.info("Sai Sri Lakshmi Gayathri Vegi | Powered by Streamlit")
