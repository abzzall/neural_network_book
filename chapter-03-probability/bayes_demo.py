"""Chapter 3: Bayes' rule for a screening test for a rare condition."""

prior = 0.01            # P(condition): 1 person in 100 has it
sensitivity = 0.90      # P(positive | condition)
false_positive = 0.05   # P(positive | no condition)

# Total probability of a positive result, over both cases.
evidence = sensitivity * prior + false_positive * (1 - prior)
# Bayes' rule: P(condition | positive).
posterior = sensitivity * prior / evidence

if __name__ == "__main__":
    print(f"P(positive) = {evidence:.4f}")
    print(f"P(condition | positive) = {posterior:.4f}")
