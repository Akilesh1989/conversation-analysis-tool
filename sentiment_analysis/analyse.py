import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/Users/akilesh/Documents/akilesh/personal/interview/topical_chat_10000.csv')

# Analysis
num_conversations = df['conversation_id'].nunique()
total_rows = len(df)
avg_msgs_per_convo = df.groupby('conversation_id').size().mean()
# Get min and max messages per conversation
msgs_per_convo = df.groupby('conversation_id').size()
min_msgs = msgs_per_convo.min()
max_msgs = msgs_per_convo.max()
min_convo_id = msgs_per_convo[msgs_per_convo == min_msgs].index[0]
max_convo_id = msgs_per_convo[msgs_per_convo == max_msgs].index[0]


# Sentiment distribution
sentiment_dist = df['sentiment'].value_counts()

# Generate report
report = f"""Topical Chat Dataset Analysis
=======================

Basic Statistics:
- Total number of conversations: {num_conversations}
- Total number of messages: {total_rows}
- Average messages per conversation: {avg_msgs_per_convo:.2f}
- Minimum messages per conversation: {min_msgs}
- Maximum messages per conversation: {max_msgs}
- Conversation with minimum messages: {min_convo_id}
- Conversation with maximum messages: {max_convo_id}

Sentiment Distribution:
{sentiment_dist.to_string()}

Data Skew Analysis:
- Messages per conversation distribution:
{df.groupby('conversation_id').size().describe().to_string()}
"""

# Create sentiment distribution plot
plt.figure(figsize=(10,6))
sentiment_dist.plot(kind='bar')
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/akilesh/Documents/akilesh/personal/interview/presentation/sentiment_analysis/sentiment_dist.png')

# Write report to file
with open('/Users/akilesh/Documents/akilesh/personal/interview/presentation/sentiment_analysis/report.txt', 'w') as f:
    f.write(report)
