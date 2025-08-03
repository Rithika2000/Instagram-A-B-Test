# Instagram Recommendation Algorithm + A/B Testing Experiment

This README provides a complete overview of how Instagram recommends Reels and posts to users based on their activity and how to design an A/B test to evaluate competing recommendation strategies.

---

## 📌 Part 1: How Instagram’s Recommendation Algorithm Works

Instagram’s algorithm is designed to keep users engaged by showing them the most relevant content based on their behavior, preferences, and platform-wide trends.

### 🔍 1. Input Signals Collected

The algorithm considers the following:

- **User Activity**:
  - Posts liked, commented, shared, saved
  - Reels watched fully or skipped
  - Search history, followed creators
- **Content Information**:
  - Type (Reel, Post), hashtags, audio, caption, visual tags
- **Creator Signals**:
  - Is_verified, follower count, past engagement rate
- **Engagement Metrics**:
  - Likes, comments, shares, saves, views
- **Embedding Representations**:
  - Each user/post is represented in a high-dimensional vector space (user_embedding, content_embedding)

### 🧠 2. Scoring & Ranking Pipeline

- **Candidate Generation**: Fetch possible posts based on user/content similarity
- **Filtering**: Remove duplicates, unsafe, stale content
- **Scoring Models**: Assign engagement probabilities (`model_score_A`, `model_score_B`)
- **Final Score**: Combine multiple scores into a `final_recommendation_score`
- **Diversity Logic**: Ensure feed variety (post type, creators)

---

## 🧪 Part 2: A/B Testing — Interest Hashtags vs. Influencers

We evaluate whether recommending trending hashtag content leads to higher engagement than featuring influencer content.

### ✅ 1. Hypothesis

- **Group A**: Posts matching top 20 trending hashtags
- **Group B**: Posts from verified or top 20% follower creators

**H₀ (Null Hypothesis)**: Engagement is equal across A and B  
**H₁ (Alternative)**: Engagement differs  
**Test Type**: Two-tailed, α = 0.05

---

### 📊 2. Data Preparation

- Load dataset from CSV
- Remove columns with 100% missing data
- Drop non-critical fields with high missing % (e.g., `location`)
- Impute:
  - `video_play_count` → 0
  - Text fields → empty string
- Drop rows with nulls in `hashtags`

### 🧹 3. Preprocessing

- Parse `hashtags` to list using `literal_eval`
- Ensure `likes`, `num_comments`, `followers` are numeric

---

### 🧪 4. Assign A/B Groups

```python
np.random.seed(42)
df['ab_group'] = np.random.choice(['A', 'B'], size=len(df))
```

---

### 🏷️ 5. Label by Strategy

- **Influencer Post**:
  - Verified OR top 20% by `followers`
- **Interest Hashtag Post**:
  - Any hashtag ∈ top 20 global hashtags (from frequency count)

```python
df['is_influencer_post'] = ...
df['matches_interest_hashtag'] = ...
```

---

### 📐 6. Calculate Engagement Metrics

- **Engagement** = `likes + num_comments`
- **Engagement Rate** = `engagement / followers`

Then filter:

```python
df_A = df[(df['ab_group'] == 'A') & (df['matches_interest_hashtag'])]
df_B = df[(df['ab_group'] == 'B') & (df['is_influencer_post'])]
```

---

### 📈 7. Welch’s t-Test

Compute t-statistic:

\[
t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}
\]

Where:
- \(\bar{x}\): Mean of engagement rate
- \(s^2\): Sample variance
- \(n\): Sample size

---

### 🧮 8. Degrees of Freedom

Use Welch–Satterthwaite formula:

\[
df = \frac{(s_A^2/n_A + s_B^2/n_B)^2}{\frac{(s_A^2/n_A)^2}{n_A-1} + \frac{(s_B^2/n_B)^2}{n_B-1}}
\]

---

### 🧪 9. p-Value Computation

- Compute area under t-distribution curve (right-tail)
- Multiply by 2 (two-tailed test)

If **p ≤ 0.05**, reject null hypothesis.

Example:

```
t = 3.46
df ≈ 6
p ≈ 0.013
```

Conclusion: **Hashtag-based feed significantly outperforms influencer feed.**

---

## 📌 Final Takeaway

We first examined how Instagram’s recommendation system works through user signals and ML scoring pipelines. Then, using a clean A/B test on real data, we discovered that interest-aligned hashtag content leads to higher engagement than influencer content.

**Recommendation**: Prioritize content aligned with trending hashtags to improve feed relevance and user interaction.

---
