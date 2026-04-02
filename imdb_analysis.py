import pandas as pd
import ast
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def safe_parse(val):
    if pd.isna(val):
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def extract_names(val, key="name"):
    items = safe_parse(val)
    return [item.get(key, "") for item in items if isinstance(item, dict)]


def extract_by_job(val, job_filter):
    items = safe_parse(val)
    return [
        item.get("name", "")
        for item in items
        if isinstance(item, dict) and item.get("job", "").lower() == job_filter.lower()
    ]


# -------------------------------------------------------
# Load data
# -------------------------------------------------------

df = pd.read_csv("imdb_data.csv")


# -------------------------------------------------------
# Section 0: Data Exploration and Sanity Check
# -------------------------------------------------------

print("Section 0: Data Exploration and Sanity Check")
print("-" * 50)

print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")

print("\nData types:")
print(df.dtypes.to_string())

print("\nMissing values (columns with at least one missing):")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"Missing Count": missing, "Missing Pct": missing_pct})
print(missing_summary[missing_summary["Missing Count"] > 0].to_string())

print("\nNumerical summary (budget, revenue, runtime, popularity):")
print(df[["budget", "revenue", "runtime", "popularity"]].describe().to_string())

zero_budget  = (df["budget"] == 0).sum()
zero_revenue = (df["revenue"] == 0).sum()
print(f"\nRows where budget is 0  : {zero_budget}")
print(f"Rows where revenue is 0 : {zero_revenue}")
print("\nNote: rows with budget = 0 are excluded from all financial calculations.")


# -------------------------------------------------------
# Prepare a clean working dataframe for financial analysis
# -------------------------------------------------------

df_fin = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
df_fin["profit"]     = df_fin["revenue"] - df_fin["budget"]
df_fin["roi"]        = (df_fin["profit"] / df_fin["budget"]) * 100
df_fin["director"]   = df_fin["crew"].apply(lambda x: extract_by_job(x, "Director"))
df_fin["producer"]   = df_fin["crew"].apply(lambda x: extract_by_job(x, "Producer"))
df_fin["actors"]     = df_fin["cast"].apply(lambda x: extract_names(x))
df_fin["genre_list"] = df_fin["genres"].apply(lambda x: extract_names(x))
df_fin["language"]   = df_fin["original_language"]

print(f"\nRows available for financial analysis (budget > 0 and revenue > 0): {len(df_fin)}")


# -------------------------------------------------------
# Question 1: Which movie made the highest profit?
#             Who were its producer and director?
#             Identify the actors in that film.
# -------------------------------------------------------

print("\n\nQuestion 1: Highest profit movie, its producer, director, and cast")
print("-" * 50)

top_movie = df_fin.loc[df_fin["profit"].idxmax()]

print(f"\nMovie      : {top_movie['title']}")
print(f"Budget     : ${top_movie['budget']:,.0f}")
print(f"Revenue    : ${top_movie['revenue']:,.0f}")
print(f"Profit     : ${top_movie['profit']:,.0f}")
print(f"ROI        : {top_movie['roi']:.1f}%")
print(f"Director   : {', '.join(top_movie['director']) if top_movie['director'] else 'N/A'}")
print(f"Producers  : {', '.join(top_movie['producer']) if top_movie['producer'] else 'N/A'}")
print(f"Top cast   : {', '.join(top_movie['actors'][:10]) if top_movie['actors'] else 'N/A'}")


# -------------------------------------------------------
# Question 2: Which language has the highest average ROI?
# -------------------------------------------------------

print("\n\nQuestion 2: Language with the highest average ROI")
print("-" * 50)

lang_roi = (
    df_fin.groupby("language")["roi"]
    .agg(avg_roi="mean", movie_count="count")
    .reset_index()
    .sort_values("avg_roi", ascending=False)
)

# Require at least 3 movies per language to keep the ranking meaningful
lang_roi_filtered = lang_roi[lang_roi["movie_count"] >= 3].reset_index(drop=True)

print("\nTop 10 languages by average ROI (minimum 3 movies each):\n")
print(f"{'Rank':<6} {'Language':<12} {'Avg ROI (%)':<20} {'Movies'}")
print("-" * 45)
for i, row in lang_roi_filtered.head(10).iterrows():
    print(f"{i+1:<6} {row['language']:<12} {row['avg_roi']:>18.1f}%   {int(row['movie_count'])}")

top_lang = lang_roi_filtered.iloc[0]
print(f"\nAnswer: Language '{top_lang['language']}' has the highest average ROI "
      f"({top_lang['avg_roi']:.1f}%) across {int(top_lang['movie_count'])} movies.")

en_row = lang_roi[lang_roi["language"] == "en"].iloc[0]
print(f"English for reference: {en_row['avg_roi']:.1f}% average ROI across {int(en_row['movie_count'])} movies.")


# -------------------------------------------------------
# Question 3: Find the unique genres in the dataset
# -------------------------------------------------------

print("\n\nQuestion 3: Unique genres in the dataset")
print("-" * 50)

all_genres = set()
for val in df["genres"].dropna():
    for item in safe_parse(val):
        if isinstance(item, dict) and "name" in item:
            all_genres.add(item["name"])

sorted_genres = sorted(all_genres)
print(f"\nTotal unique genres: {len(sorted_genres)}\n")
for i, genre in enumerate(sorted_genres, 1):
    print(f"  {i:>2}. {genre}")

genre_counts = {}
for val in df["genres"].dropna():
    for item in safe_parse(val):
        if isinstance(item, dict):
            g = item.get("name", "")
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1

print("\nGenre frequency across all 3000 movies:\n")
print(f"  {'Genre':<30} {'Count':>6}")
print("  " + "-" * 38)
for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
    print(f"  {genre:<30} {count:>6}")


# -------------------------------------------------------
# Question 4: Table of all producers and directors.
#             Top 3 producers by highest average ROI.
# -------------------------------------------------------

print("\n\nQuestion 4: Producers and directors per movie, and top 3 producers by average ROI")
print("-" * 50)

rows = []
for _, row in df_fin.iterrows():
    rows.append({
        "Title":    row["title"],
        "Director": ", ".join(row["director"]) if row["director"] else "N/A",
        "Producer": ", ".join(row["producer"]) if row["producer"] else "N/A",
        "Budget":   row["budget"],
        "Revenue":  row["revenue"],
        "Profit":   row["profit"],
        "ROI (%)":  round(row["roi"], 1),
    })

prod_dir_df = pd.DataFrame(rows)

print("\nProducers and Directors table (first 20 rows):\n")
print(prod_dir_df.head(20).to_string(index=False))

exploded = []
for _, row in df_fin.iterrows():
    for p in row["producer"]:
        if p:
            exploded.append({"producer": p, "roi": row["roi"]})

prod_exp = pd.DataFrame(exploded)
top3_producers = (
    prod_exp.groupby("producer")["roi"]
    .agg(avg_roi="mean", movies_produced="count")
    .reset_index()
    .query("movies_produced >= 2")
    .sort_values("avg_roi", ascending=False)
    .head(3)
    .reset_index(drop=True)
)

print("\nTop 3 producers by average ROI (minimum 2 movies produced):\n")
print(f"{'Rank':<6} {'Producer':<35} {'Avg ROI (%)':<20} {'Movies Produced'}")
print("-" * 70)
for i, row in top3_producers.iterrows():
    print(f"{i+1:<6} {row['producer']:<35} {row['avg_roi']:>18.1f}%   {int(row['movies_produced'])}")


# -------------------------------------------------------
# Question 5: Which actor has acted in the most movies?
#             Deep dive into their movies, genres, profits.
# -------------------------------------------------------

print("\n\nQuestion 5: Most prolific actor and deep dive analysis")
print("-" * 50)

actor_movies = {}
for _, row in df_fin.iterrows():
    for actor in row["actors"]:
        if actor:
            actor_movies.setdefault(actor, []).append(row)

actor_count = {k: len(v) for k, v in actor_movies.items()}
top_actor   = max(actor_count, key=actor_count.get)
actor_df    = pd.DataFrame(actor_movies[top_actor])

print(f"\nActor with most movies: {top_actor} ({actor_count[top_actor]} movies)\n")

print(f"Movies featuring {top_actor} (sorted by profit):\n")
print(f"  {'Title':<50} {'Genres':<35} {'Profit':>15}  {'ROI (%)':>8}")
print("  " + "-" * 115)
for _, m in actor_df.sort_values("profit", ascending=False).iterrows():
    genre_str = ", ".join(m["genre_list"][:3]) if m["genre_list"] else "N/A"
    print(f"  {m['title'][:49]:<50} {genre_str[:34]:<35} "
          f"${m['profit']:>14,.0f}  {m['roi']:>7.1f}%")

print(f"\nFinancial summary for {top_actor}:")
print(f"  Total profit across all films : ${actor_df['profit'].sum():,.0f}")
print(f"  Average ROI                   : {actor_df['roi'].mean():.1f}%")
print(f"  Best performing movie         : {actor_df.loc[actor_df['profit'].idxmax(), 'title']}")
print(f"  Worst performing movie        : {actor_df.loc[actor_df['profit'].idxmin(), 'title']}")

genre_dist = {}
for _, m in actor_df.iterrows():
    for g in m["genre_list"]:
        genre_dist[g] = genre_dist.get(g, 0) + 1

print(f"\nGenre distribution for {top_actor}:\n")
print(f"  {'Genre':<30} {'Movies':>6}")
print("  " + "-" * 38)
for genre, cnt in sorted(genre_dist.items(), key=lambda x: -x[1]):
    print(f"  {genre:<30} {cnt:>6}")


# -------------------------------------------------------
# Question 6: Which actors do the top 3 directors
#             prefer the most?
# -------------------------------------------------------

print("\n\nQuestion 6: Top 3 directors and their preferred actors")
print("-" * 50)

director_movies = {}
for _, row in df_fin.iterrows():
    for d in row["director"]:
        if d:
            director_movies.setdefault(d, []).append(row)

dir_count = {k: len(v) for k, v in director_movies.items()}
top3_dirs = sorted(
    [(d, c) for d, c in dir_count.items() if c >= 3],
    key=lambda x: -x[1]
)[:3]

for rank, (director, num_movies) in enumerate(top3_dirs, 1):
    movies     = pd.DataFrame(director_movies[director])
    actor_freq = {}
    for _, row in movies.iterrows():
        for actor in row["actors"]:
            if actor:
                actor_freq[actor] = actor_freq.get(actor, 0) + 1

    top_actors = sorted(actor_freq.items(), key=lambda x: -x[1])[:8]

    print(f"\n  #{rank} {director} ({num_movies} movies in dataset)")
    print(f"  Films: {', '.join(movies['title'].tolist())}\n")
    print(f"  {'Actor':<35} {'Collaborations':>14}")
    print("  " + "-" * 52)
    for actor, collab in top_actors:
        print(f"  {actor:<35} {collab:>14}")


print("\n\nAnalysis complete.")
