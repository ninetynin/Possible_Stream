import matplotlib.pyplot as plt
import pandas as pd

def plot_age_distribution(df):
    plt.figure(figsize=(10, 5))
    plt.hist(df['age'].dropna(), bins=range(0, 101, 5), edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_emotion_distribution(df):
    plt.figure(figsize=(10, 5))
    emotion_counts = df['emotion'].value_counts()
    plt.bar(emotion_counts.index, emotion_counts.values)
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def main():
    df = pd.read_sql_query('SELECT * FROM analysis', 'sqlite:///db/analysis_results.db')
    plot_age_distribution(df)
    plot_emotion_distribution(df)

if __name__ == '__main__':
    main()
