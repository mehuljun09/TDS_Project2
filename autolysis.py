import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Ensure AI Proxy Token is set
if "AIPROXY_TOKEN" not in os.environ:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
openai.api_key = eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDE1NDlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.rUb9YP_q3fP3uQ2opgaGofyjRt6Vm1oXf4qtOOUi8Eg

# Function to generate AI story using GPT-4o-Mini
def generate_story(analysis_summary, top_features):
    prompt = f"""
    You are a data analyst. You have performed an analysis on a dataset and here are the results:

    Summary:
    {analysis_summary}

    Key insights include:
    {top_features}

    Write a Markdown report titled 'README.md' that tells a compelling story about the analysis. 
    Include descriptions of up to three charts and provide insights based on the data.
    """

    response = openai.Completion.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Function to perform automated analysis
def analyze_data(df):
    analysis_summary = df.describe().to_string()
    print("Generated basic statistics for the dataset.")

    # Extract top features
    top_features = df.columns[:3].tolist()
    print("Top features identified:", top_features)

    return analysis_summary, top_features

# Function to generate charts
def generate_charts(df, output_prefix="chart"):
    charts = []

    # First Chart: Pairplot
    sns.pairplot(df.select_dtypes(include=['number']).sample(n=100, random_state=42))
    pairplot_file = f"{output_prefix}_pairplot.png"
    plt.savefig(pairplot_file)
    plt.close()
    charts.append(pairplot_file)
    print(f"Generated chart: {pairplot_file}")

    # Second Chart: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    heatmap_file = f"{output_prefix}_heatmap.png"
    plt.savefig(heatmap_file)
    plt.close()
    charts.append(heatmap_file)
    print(f"Generated chart: {heatmap_file}")

    # Third Chart: Histogram for the first numerical column
    num_column = df.select_dtypes(include=['number']).columns[0]
    plt.figure(figsize=(8, 6))
    sns.histplot(df[num_column], bins=30, kde=True)
    hist_file = f"{output_prefix}_histogram.png"
    plt.savefig(hist_file)
    plt.close()
    charts.append(hist_file)
    print(f"Generated chart: {hist_file}")

    return charts

# Function to write README.md
def write_markdown(story, charts):
    with open("README.md", "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write(story + "\n\n")
        f.write("## Supporting Visualizations\n")
        for chart in charts:
            f.write(f"![Chart](./{chart})\n\n")
    print("README.md file has been created.")

# Main script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset: {input_file}")
    df = pd.read_csv(input_file)

    # Perform analysis
    analysis_summary, top_features = analyze_data(df)

    # Generate charts
    charts = generate_charts(df)

    # Generate story using AI
    print("Generating analysis story with GPT-4o-Mini...")
    story = generate_story(analysis_summary, top_features)

    # Write results to README.md
    write_markdown(story, charts)
    print("Analysis completed successfully!")
