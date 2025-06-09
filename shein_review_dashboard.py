import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt
import review_tools as rt
import glob
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Shein App Review Insights", layout="wide")



# --- HEADER ---

# Create two columns: one for the logo (0.1 fraction of width), one for the text (0.9 fraction)
col1, col2 = st.columns([0.1, 0.9])

with col1:
    # Assuming 'shein_logo.png' is in the same folder as this script
    st.image("shein_logo.png", width=100) # Adjust width as needed for inline placement

with col2:
    st.markdown("## Shein App Review Dashboard")

st.markdown("Explore customer sentiment, delivery issues, and product perceptions via real user reviews.")

# ---LIVE UPDATE---
if st.sidebar.button("🔄 Refresh Reviews"):
    with st.spinner("Updating and analyzing reviews..."):
        rt.update_reviews_pipeline() # Assuming rt.update_reviews() should be rt.update_reviews_pipeline() based on previous discussions
    st.success("Reviews updated!")

# --- LOAD LATEST REVIEW FILE ---
def load_latest_file():
    files = glob.glob("updated_reviews_*.csv")
    if not files:
        st.error("No review files found.")
        return None
    latest_file = max(files, key=os.path.getctime)
    return pd.read_csv(latest_file, parse_dates=["at"])

reviews_df = load_latest_file()
if reviews_df is not None:
    st.success(f"Loaded {len(reviews_df)} reviews from latest file.")
    last_updated = pd.to_datetime(reviews_df['at']).max().strftime('%Y-%m-%d %H:%M')
    st.sidebar.caption(f"📅 Last update: {last_updated}")

    # --- SIDEBAR FILTERS WITH "SELECT ALL" TOGGLE ---
    st.sidebar.header("🔎 Filter Reviews")

    # Date Range
    date_range = st.sidebar.date_input("Date Range", [reviews_df["at"].min().date(), reviews_df["at"].max().date()])

    # Sentiment Filter
    all_sentiments = reviews_df["sentiment"].unique()
    select_all_sentiments = st.sidebar.checkbox("Select All Sentiments", value=True)
    sentiments = all_sentiments if select_all_sentiments else st.sidebar.multiselect("Choose Sentiments", all_sentiments)

    # Theme Filter
    all_themes = reviews_df["theme"].unique()
    select_all_themes = st.sidebar.checkbox("Select All Themes", value=True)
    themes = all_themes if select_all_themes else st.sidebar.multiselect("Choose Themes", all_themes)

    # --- APPLY FILTERS ---
    filtered_reviews = reviews_df[
        (reviews_df["at"].dt.date >= date_range[0]) &
        (reviews_df["at"].dt.date <= date_range[1]) &
        (reviews_df["sentiment"].isin(sentiments)) &
        (reviews_df["theme"].isin(themes))
    ]

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Distribution", "💬 Review Samples", "📆 Trend", "☁️ Word Cloud", "📋 Sentiment Count"])

    # --- TAB 1: BAR CHART ---
    with tab1:
        st.markdown("### 📊 Sentiment Distribution by Theme")

        sentiment_theme = filtered_reviews.groupby(['theme', 'sentiment']).size().unstack().fillna(0)
        sentiment_theme_reset = sentiment_theme.reset_index().melt(id_vars="theme", var_name="Sentiment", value_name="Count")

        bar_chart = alt.Chart(sentiment_theme_reset).mark_bar().encode(
            x=alt.X("theme:N", title="Theme"),
            y=alt.Y("Count:Q"),
            color=alt.Color("Sentiment:N", scale=alt.Scale(scheme="blues")),
            tooltip=["theme", "Sentiment", "Count"]
        ).properties(width=800, height=400)

        st.altair_chart(bar_chart, use_container_width=True)

    # --- TAB 2: REVIEW SAMPLES ---
    with tab2:
        st.markdown("### 💬 Sample Reviews by Theme")
        # Ensure that this selectbox uses the unique themes from filtered_reviews or reviews_df
        # If filtered_reviews is empty, reviews_df.unique() is safer to populate the selectbox
        theme_options = filtered_reviews["theme"].unique() if not filtered_reviews.empty else reviews_df["theme"].unique()
        
        if len(theme_options) > 0:
            theme_selected = st.selectbox("Choose a Theme", theme_options)

            # Ensure there are reviews for the selected theme before attempting to sample
            reviews_for_sampling = filtered_reviews[filtered_reviews["theme"] == theme_selected]

            if not reviews_for_sampling.empty:
                # Use min(5 to 8, len(reviews_for_sampling)) to prevent ValueError if less than 5 samples are available
                num_samples = min(8, len(reviews_for_sampling))
                sample_reviews = reviews_for_sampling.sample(num_samples, random_state=42) # Added random_state for reproducibility
                
                for _, row in sample_reviews.iterrows():
                    st.markdown(f"**Sentiment:** {row['sentiment']} | **Rating:** {row['score']}")
                    st.write(f"> {row['content']}")
                    st.markdown("---")
            else:
                st.info(f"No reviews found for the selected theme: **{theme_selected}** within the current filters.")
        else:
            st.info("No themes available after current filters. Please adjust filters.")

    # --- TAB 3: LINE CHART ---
    with tab3:
        st.markdown("### 📆 Sentiment Trend Over Time")

        # Check if filtered_reviews is empty to prevent errors
        if not filtered_reviews.empty:
            sentiment_timeline = filtered_reviews.groupby([filtered_reviews["at"].dt.date, "sentiment"]).size().unstack().fillna(0)
            timeline_melted = sentiment_timeline.reset_index().melt(id_vars="at", var_name="Sentiment", value_name="Count")

            line_chart = alt.Chart(timeline_melted).mark_line(point=True).encode(
                x="at:T",
                y="Count:Q",
                color=alt.Color("Sentiment:N", scale=alt.Scale(scheme="blues")),
                tooltip=["at", "Sentiment", "Count"]
            ).properties(width=800, height=400)

            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.info("No data available for sentiment trend with current filters.")


    # --- TAB 4: WORD CLOUD ---
    with tab4:
        st.markdown("### ☁️ Word Cloud by Theme")
        
        theme_wc_options = filtered_reviews["theme"].unique() if not filtered_reviews.empty else reviews_df["theme"].unique()

        if len(theme_wc_options) > 0:
            theme_wc = st.selectbox("Choose a Theme for WordCloud", theme_wc_options, key="wc")
            text_data = " ".join(filtered_reviews[filtered_reviews["theme"] == theme_wc]["content"].dropna().astype(str))

            if text_data.strip():
                # Corrected use_container_width for st.image
                wc = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text_data)
                st.image(wc.to_array(), use_container_width=True) # Changed to use_column_width as use_container_width is not supported for st.image directly with numpy arrays.
            else:
                st.write("No reviews available for this theme in selected filters.")
        else:
            st.info("No themes available for Word Cloud with current filters.")


    # --- TAB 5: 📋 Sentiment Distribution Table ---
    with tab5: # This was correctly indented in your original
        st.markdown("### 📑 Sentiment Count by Theme")

        dist_df = filtered_reviews.groupby(['theme', 'sentiment']).size().reset_index(name='count')

        # Define a list of all possible sentiment columns that *should* exist
        all_sentiment_cols = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

        # Use pivot_table for robustness in creating columns
        dist_pivot = pd.pivot_table(dist_df, values='count', index='theme',
                                    columns='sentiment', fill_value=0).astype(int)

        # Ensure all expected sentiment columns are present (reindex after pivot_table for extra safety)
        dist_pivot = dist_pivot.reindex(columns=all_sentiment_cols, fill_value=0)

        # Sort by 'NEGATIVE' if it exists and has non-zero values, otherwise sort by theme name
        if 'NEGATIVE' in dist_pivot.columns and dist_pivot['NEGATIVE'].sum() > 0:
            dist_pivot = dist_pivot.sort_values(by='NEGATIVE', ascending=False)
        else:
            # Fallback sort if 'NEGATIVE' column is all zeros or not present
            dist_pivot = dist_pivot.sort_index(ascending=True) # Sort by theme name (index)

        def highlight_sentiment(val, sentiment):
            if sentiment == 'POSITIVE':
                return 'background-color: #d4edda; color: #155724'
            elif sentiment == 'NEGATIVE':
                return 'background-color: #f8d7da; color: #721c24'
            elif sentiment == 'NEUTRAL':
                return 'background-color: #e2e3e5; color: #383d41'
            return ''

        styled = dist_pivot.style.set_caption("Sentiment Counts per Theme")
        for col in dist_pivot.columns:
            styled = styled.applymap(lambda v, s=col: highlight_sentiment(v, s), subset=[col])

        st.dataframe(styled, use_container_width=True)

    # --- DOWNLOAD BUTTON ---
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📁 Download Filtered Data",
        data=filtered_reviews.to_csv(index=False).encode("utf-8"),
        file_name="filtered_shein_reviews.csv",
        mime="text/csv"
    )

else:
    st.info("Please load reviews to display the dashboard.")

# --- FOOTER (Place this at the very end of your shein_review_dashboard.py file) ---
# Add a horizontal line for separation if you like
st.markdown("---")
st.markdown(
    """
    <div style="font-size: 0.8em; color: grey; text-align: center;">
    Shein logo icon used with thanks to  <a href="https://icons8.com/icon/V3r2kWDPwZgQ/shein" target="_blank" style="color: grey; text-decoration: none;">Icons8 under free license.</a>
    </div>
    """,
    unsafe_allow_html=True # Required to render HTML tags like <div> and <a>
)