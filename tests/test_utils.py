# This file will contain unit tests for utility functions in utils.py
import pytest
from doc_crawler.utils import is_doc, chunk_content #, embed_chunks, store_in_qdrant, get_urls, find_docs

# Basic HTML content samples
doc_html_sample_keywords = """
<html><head><title>My Doc Page</title></head>
<body><h1>Welcome to My Documentation</h1>
<p>This guide helps you understand our API. Check the reference section.</p>
<pre><code>print('Hello World')</code></pre>
</body></html>
"""

doc_html_sample_code = """
<html><head><title>Code Examples</title></head>
<body><h1>Code Snippets</h1>
<p>Here is an example:</p>
<pre><code>def example():\n    return True</code></pre>
</body></html>
"""

non_doc_html_sample = """
<html><head><title>My Blog Post</title></head>
<body><h1>Latest News</h1>
<p>Today, something interesting happened. Not related to docs.</p>
<img src="image.jpg" alt="A random image">
</body></html>
"""

empty_html_sample = ""

markdown_sample_long_paragraphs = """
This is the first paragraph. It is quite long and contains multiple sentences. We are testing the paragraph splitting strategy.

This is the second paragraph. It also has several sentences. The goal is to see if these are correctly identified as separate chunks, or combined if they are short enough and the next one fits.

This is a third, much shorter paragraph.

And a fourth one.
"""

markdown_sample_sentences = """
First sentence here. Second sentence follows. Is this a third sentence? Yes, it is!
A new line, but still part of the previous conceptual block if splitting by paragraph.
However, if splitting by sentence, this should be multiple chunks.
"""

class TestIsDoc:
    def test_is_doc_with_keywords(self):
        assert is_doc(doc_html_sample_keywords) == True

    def test_is_doc_with_code_elements(self):
        assert is_doc(doc_html_sample_code) == True

    def test_is_doc_non_doc_content(self):
        assert is_doc(non_doc_html_sample) == False

    def test_is_doc_empty_content(self):
        assert is_doc(empty_html_sample) == False

    def test_is_doc_minimal_doc_signature(self):
        minimal_doc = "<h1>API Reference</h1> <p>Details about endpoints.</p>"
        assert is_doc(minimal_doc) == True

    def test_is_doc_ambiguous_but_has_code(self):
        ambiguous_content = "<h1>A Page</h1> <p>Some text.</p> <code>print(1)</code>"
        assert is_doc(ambiguous_content) == True

class TestChunkContent:
    def test_chunk_by_paragraph_long(self):
        chunks = chunk_content(markdown_sample_long_paragraphs, strategy="paragraph", chunk_size=100)
        # Expected: Each paragraph might be its own chunk if > chunk_size or combined if they fit.
        # Given chunk_size 100, the first two paragraphs are likely separate.
        # The third and fourth might combine or be separate.
        # This heuristic is simple, so exact numbers depend on whitespace and combined length.
        # print(f"Paragraph Chunks: {chunks}") # For debugging
        assert len(chunks) >= 2 # Expecting at least the first two long paragraphs to be chunks
        assert "This is the first paragraph." in chunks[0]
        assert "This is the second paragraph." in chunks[1]
        # Check if shorter paragraphs are combined or separate based on the simple logic
        if len(chunks) > 2:
            if "This is a third, much shorter paragraph." in chunks[2] and "And a fourth one." in chunks[2]:
                 assert "This is a third, much shorter paragraph.\n\nAnd a fourth one." == chunks[2]
            elif "This is a third, much shorter paragraph." in chunks[2]:
                 assert "This is a third, much shorter paragraph." == chunks[2]
                 if len(chunks) > 3:
                    assert "And a fourth one." == chunks[3]


    def test_chunk_by_paragraph_short_combined(self):
        short_md = "Short para1.\n\nShort para2.\n\nShort para3."
        chunks = chunk_content(short_md, strategy="paragraph", chunk_size=200)
        assert len(chunks) == 1
        assert chunks[0] == "Short para1.\n\nShort para2.\n\nShort para3."

    def test_chunk_by_sentence(self):
        chunks = chunk_content(markdown_sample_sentences, strategy="sentence", chunk_size=50)
        # print(f"Sentence Chunks: {chunks}") # For debugging
        assert len(chunks) > 1
        assert "First sentence here." in chunks[0]
        assert "Second sentence follows." in chunks[0] or "Second sentence follows." in chunks[1] # depends on exact length
        assert "Yes, it is!" in chunks[0] or "Yes, it is!" in chunks[1] # depends on exact length
        # Check that sentences are not split mid-way (roughly)
        for chunk in chunks:
            assert len(chunk) <= 50 or (len(chunk) > 50 and chunk.count(" ") == 0) # allow oversized if single word

    def test_chunk_by_sentence_long_sentences(self):
        long_sentence_md = "This is a very long first sentence that should definitely exceed the chunk size and be its own chunk. This is another very long second sentence that also should be its own chunk because of the length."
        chunks = chunk_content(long_sentence_md, strategy="sentence", chunk_size=60)
        # print(f"Long Sentence Chunks: {chunks}")
        assert len(chunks) == 2
        assert chunks[0] == "This is a very long first sentence that should definitely exceed the chunk size and be its own chunk."
        assert chunks[1] == "This is another very long second sentence that also should be its own chunk because of the length."


    def test_chunk_with_default_strategy_fixed_size(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 2 # 52 chars
        chunks = chunk_content(text, strategy="unknown", chunk_size=20, overlap=5)
        # Expects chunks of 20, with 5 overlap
        # 1. text[0:20] = "abcdefghijklmnopqrst"
        # 2. text[15:35] = "pqrstuvwxyzabcde"
        # 3. text[30:50] = "efghijklmnopqrstuvwxyz"
        # 4. text[45:52] = "uvwxyz" (remaining part)
        assert len(chunks) == 4
        assert chunks[0] == "abcdefghijklmnopqrst"
        assert chunks[1] == "pqrstuvwxyzabcde" # 20 chars from index 15
        assert chunks[2] == "efghijklmnopqrstuvwxyz" # 20 chars from index 30
        assert chunks[3] == "uvwxyzab" # Corrected: last chunk is text[45:65] but text ends at 52. So text[45:52+5] -> text[45:57] which is text[45:]
                                      # Actually, my calculation for the last chunk was off.
                                      # text[0:20]
                                      # text[15:35]
                                      # text[30:50]
                                      # text[45:65] -> text[45:52] "uvwxyzab" is wrong.
                                      # it should be text[45:len(text)] which is text[45:52] = "uvwxyz"
        # The default fixed size chunker is basic:
        # chunk1 = text[0:20]
        # chunk2 = text[15:35] (20-5 = 15)
        # chunk3 = text[30:50] (15+15 = 30)
        # chunk4 = text[45:65] -> text[45:52]
        # For "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
        # c1: "abcdefghijklmnopqrst"
        # c2: "pqrstuvwxyzabcde"
        # c3: "efghijklmnopqrstuvwxyz"
        # c4: "uvwxyz" (length 7)
        # Let's re-verify the default chunker logic in utils.py
        # `for i in range(0, len(markdown_content), chunk_size - overlap):`
        # `    chunks.append(markdown_content[i:i + chunk_size])`
        # i=0: text[0 : 0+20] -> "abcdefghijklmnopqrst"
        # i=15: text[15 : 15+20] -> "pqrstuvwxyzabcde"
        # i=30: text[30 : 30+20] -> "efghijklmnopqrstuvwxyz"
        # i=45: text[45 : 45+20] -> text[45:65], actual slice is text[45:52] -> "uvwxyz"
        assert chunks[3] == "uvwxyz"


    def test_empty_content_chunking(self):
        chunks = chunk_content("", strategy="paragraph", chunk_size=100)
        assert len(chunks) == 0
        chunks = chunk_content("", strategy="sentence", chunk_size=100)
        assert len(chunks) == 0
        chunks = chunk_content("", strategy="fixed", chunk_size=100)
        assert len(chunks) == 0

# More tests to be added for embed_chunks (mocking Ollama), store_in_qdrant (mocking Qdrant),
# get_urls (mocking crawl4ai), and find_docs.
# For now, focusing on is_doc and chunk_content as they don't have external dependencies.

# Example of how to run tests:
# Ensure pytest is installed: pip install pytest
# Navigate to the root directory of the project (above 'doc_crawler' and 'tests')
# Run: pytest
# Or: python -m pytest
# Or: pytest tests/test_utils.py
