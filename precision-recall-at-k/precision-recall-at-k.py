def precision_recall_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    recommended_set = set(recommended_at_k)
    relevant_set = set(relevant)

    true_positives = len(recommended_set & relevant_set)

    precision = true_positives / k if k > 0 else 0.0
    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0.0

    return [precision, recall]   # ✅ return list