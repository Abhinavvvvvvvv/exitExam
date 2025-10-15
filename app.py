import streamlit as st
import numpy as np

st.title("Amazon Review Sentiment Analysis")

# Set this according to your best model: 'tfidf', 'word2vec', or 'rnn'
MODEL_TYPE = 'tfidf'  # <-- change as needed

if MODEL_TYPE == 'tfidf':
    import joblib
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    clf = joblib.load('tfidf_model.pkl')
    def predict_sentiment(text):
        x = vectorizer.transform([text])
        pred = clf.predict(x)[0]
        return 'Positive' if pred == 1 else 'Negative'

elif MODEL_TYPE == 'word2vec':
    import joblib
    import gensim
    w2v = gensim.models.Word2Vec.load('word2vec.model')
    clf = joblib.load('word2vec_rf_model.pkl')
    vocab = set(w2v.wv.index_to_key)
    def avg_word_vec(words, model, vocab, num_features):
        feature_vec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        for word in words:
            if word in vocab:
                nwords += 1
                feature_vec = np.add(feature_vec, model[word])
        if nwords > 0:
            feature_vec = np.divide(feature_vec, nwords)
        return feature_vec
    def predict_sentiment(text):
        words = text.lower().split()
        vec = avg_word_vec(words, w2v.wv, vocab, 100).reshape(1, -1)
        pred = clf.predict(vec)[0]
        return 'Positive' if pred == 1 else 'Negative'

elif MODEL_TYPE == 'rnn':
    import pickle
    from keras.models import load_model
    from keras.preprocessing.sequence import pad_sequences
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model('rnn_model.h5')
    maxlen = 100
    def predict_sentiment(text):
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=maxlen)
        pred = (model.predict(pad) > 0.5).astype('int32')[0][0]
        return 'Positive' if pred == 1 else 'Negative'

user_input = st.text_area("Enter a review for sentiment analysis:")
if st.button("Predict"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.write(f"Sentiment: **{sentiment}**")
    else:
        st.write("Please enter a review.")