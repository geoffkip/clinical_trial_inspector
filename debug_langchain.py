import langchain
import os
import sys

print(f"LangChain version: {langchain.__version__}")
print(f"LangChain path: {langchain.__path__}")
print(f"LangChain dir: {dir(langchain)}")

try:
    import langchain.chains
    print("Successfully imported langchain.chains")
except ImportError as e:
    print(f"Failed to import langchain.chains: {e}")

# Check site-packages
import site
print(f"Site packages: {site.getsitepackages()}")

