IMDB Movie Dataset Analysis
Name: Nitesh Naudiyal
Roll No.: 2472031
Subject: Introduction to Data Science
Overview
Analysis of 3000 movies from the IMDB dataset to help a rookie movie producer make data-driven decisions about what type of movies to produce and which actors to cast.
Questions Answered
Which movie made the highest profit, and who were its producer, director, and cast?
Which language has the highest average ROI?
What are the unique genres in the dataset?
Who are the top 3 producers by average ROI?
Which actor has acted in the most movies, and how have those movies performed?
Which actors do the top 3 directors prefer the most?
Files
File
Description
imdb_analysis.py
Main analysis script
imdb_data.csv
Dataset with 3000 movies
How to Run
pip install pandas
python imdb_analysis.py
Dataset
The dataset contains 3000 movies with columns including budget, revenue, cast, crew, genres, release date, and original language. Rows with missing budget values (812 records) are excluded from all financial calculations.
