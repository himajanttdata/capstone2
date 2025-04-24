import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunking:
    def chunk_code(self, code, chunk_size=512, overlap=100):
        chunks = []
        pattern = re.compile(r"(class\s+\w+\s*\(.*?\):|def\s+\w+\s*\(.*?\):)", re.MULTILINE)
        matches = list(pattern.finditer(code))

        prev_end = 0
        for match in matches:
            start = match.start()
            if start > prev_end:
                chunks.append(code[prev_end:start].strip())

            end = match.end()
            chunks.append(code[start:end].strip())
            prev_end = end

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        remaining_text = code[prev_end:].strip()
        if remaining_text:
            chunks.extend(text_splitter.split_text(remaining_text))

        return chunks
