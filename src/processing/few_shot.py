# %%
import polars as pl
import altair as alt
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
    pipeline,
)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# %%
# Fine tune model for later use in classification

# Load the data
df = pl.read_csv("./../../data/v7/front_page_data.csv", ignore_errors=True)
df2 = pl.read_csv("./../../data/regression/data.csv", ignore_errors=True)

# Prepare data of successful posts for training
df2 = df2.head(3000).with_columns(
    pl.when(pl.col("id").is_in(df["id"])).then(1).otherwise(0).alias("in_front_page")
)

X = df2["title"].to_list()
y = df2["in_front_page"].to_list()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Prepare labels
id2label = {
    0: "does not gratifiy intellectual curiosity",
    1: "gratifies intellectual curiosity",
}
label2id = {v: k for k, v in id2label.items()}

# Prepare the model & tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, label2id=label2id, id2label=id2label
)
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased", label2id=label2id, id2label=id2label
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Finalize data set
train_data = [
    {
        "input_ids": tokenizer(X_train[i], truncation=True)["input_ids"],
        "attention_mask": tokenizer(X_train[i], truncation=True)["attention_mask"],
        "label": y_train[i],
    }
    for i in range(len(X_train))
]
test_data = [
    {
        "input_ids": tokenizer(X_test[i], truncation=True)["input_ids"],
        "attention_mask": tokenizer(X_test[i], truncation=True)["attention_mask"],
        "label": y_test[i],
    }
    for i in range(len(X_test))
]


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# %%
# Use the model to classify the data

df = pl.read_csv("./../../data/regression/data.csv", ignore_errors=True)
titles = df["title"].to_list()

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
results = [classifier(doc) for doc in titles[len(titles) - 10000 :]]

# %%
alt.data_transformers.enable("vegafusion")

print(results[0])

one_hot = [
    1 if result[0]["label"] == "gratifies intellectual curiosity" else 0
    for result in results
]


df_filtered = df.tail(10000).with_columns(
    pl.Series("gratifies_intellectual_curiosity", one_hot)
)

# Display the relation of score and gratifies intellectual curiosity
fig = (
    alt.Chart(
        df_filtered.filter(pl.col("gratifies_intellectual_curiosity").is_not_null()),
        title="gratifies_curiosity",
    )
    .mark_boxplot(extent="min-max")
    .encode(
        y="gratifies_intellectual_curiosity:N",
        x=alt.X("score:Q", scale=alt.Scale(type="log"), title="Score (log scale)"),
    )
)
fig

# %%
df_filtered.write_csv("./../../data/archive/zero_shot_classification_extended_v7_.csv")

# --> This is not a feasible characteristic of posts
