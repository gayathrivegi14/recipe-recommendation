# 🍳 Healthy Recipe Recommender

A smart recipe recommendation system built with Streamlit that helps you find delicious recipes based on ingredients you have at home and your nutrition goals.

## Features

- **Ingredient-Based Search**: Enter ingredients you have, get matching recipes
- **Smart Matching**: Uses TF-IDF and cosine similarity for intelligent recipe matching
- **Nutrition Filtering**: Filter recipes by calories, fat, carbs, and protein
- **Dietary Preferences**: Support for Vegetarian, Vegan, and Gluten-Free diets
- **Beautiful UI**: Clean, intuitive interface with recipe images
- **Detailed Recipe View**: Full ingredients, directions, and nutrition info

## Installation

1. Install required packages:
```bash
pip install streamlit pandas numpy scikit-learn
```

2. Make sure you have `recipes.csv` in the same directory as `app.py`

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Enter Ingredients**: Type ingredients you have (e.g., "chicken, rice, broccoli")
2. **Adjust Settings**: 
   - Number of recommendations (3-20)
   - Match sensitivity (how closely ingredients should match)
3. **Set Nutrition Goals** (Optional):
   - Enable nutrition filtering
   - Set max calories, fat, carbs
   - Set minimum protein
4. **Choose Dietary Preferences** (Optional):
   - Vegetarian
   - Vegan
   - Gluten-Free
5. **Click "Find Recipes!"** to get personalized recommendations

## How It Works

### Recommendation Algorithm

1. **Text Processing**: Cleans and normalizes ingredient text
2. **TF-IDF Vectorization**: Converts ingredients to numerical vectors
3. **Cosine Similarity**: Calculates similarity between your ingredients and recipes
4. **Filtering**: Applies nutrition and dietary filters
5. **Ranking**: Returns top matches sorted by similarity score

### Key Components

- **RecipeRecommender Class**: Core recommendation engine
- **Nutrition Parser**: Extracts nutrition data from recipe text
- **Ingredient Cleaner**: Normalizes ingredient text for better matching
- **Streamlit UI**: Interactive web interface

## Data Format

The app expects `recipes.csv` with these columns:
- `recipe_name`: Name of the recipe
- `ingredients`: Comma-separated ingredient list
- `directions`: Cooking instructions
- `prep_time`, `cook_time`, `total_time`: Time in minutes
- `servings`: Number of servings
- `rating`: Recipe rating (0-5)
- `nutrition`: Nutrition information string
- `img_src`: Recipe image URL
- `url`: Original recipe URL
- `cuisine_path`: Cuisine category

## Tips for Best Results

- **Be specific**: "chicken breast" works better than just "chicken"
- **Use common names**: "tomato" instead of "roma tomato"
- **Multiple ingredients**: More ingredients = better matches
- **Adjust sensitivity**: Lower sensitivity (0.05-0.15) for more results
- **Nutrition filters**: Only enable if you have specific dietary goals

## Troubleshooting

**No recipes found?**
- Try lowering the match sensitivity
- Use fewer or more common ingredients
- Disable nutrition filters temporarily

**Images not loading?**
- Some recipes may not have images
- Check your internet connection

**Slow performance?**
- The first run loads and processes the dataset (cached after)
- Large datasets may take a few seconds

## Future Enhancements

- Meal planning features
- Shopping list generation
- Save favorite recipes
- User ratings and reviews
- More dietary preferences
- Cuisine-based filtering
- Cooking difficulty levels

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [Pandas](https://pandas.pydata.org/) - Data processing

---

Enjoy cooking! 🍽️
