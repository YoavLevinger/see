# see

This project aims to solve "Software Effort Estimation"(SEE) in an automated manner.


## Install the langchain-core library by running:
```
pip install langchain-core
```

If running on ubuntu, create "run environment" and then install the langchain-core:

```
python3 -m venv myenv
source myenv/bin/activate
```

To exit the virtual environment, simply run:

```
deactivate
```




Or simply install:

'''
pip install langchain
'''


----------------

The primary [LangChain GitHub repository](https://github.com/langchain-ai/langchain) hosts several PyPI packages:  
- **langchain-community**: Includes various third-party integrations.
- **langchain**: Serves as the fundamental building block of the LangChain framework.  
- **langchain-core**: Provides abstract classes that define the base interfaces for integrations.  

The **langchain-cli** package provides a command-line interface (CLI) for LangChain.  

- **langchain-text-splitters**: Offers utilities for breaking various text documents into chunks.  
- **langchain-experimental**: Contains experimental LangChain code that may later be incorporated into the core libraries.  
- **Partner libraries**: Include integrations with LangChain. Many of these libraries are hosted in separate repositories within the LangChain GitHub organization (e.g., the **langchain-google-vertexai** library is found in the **langchain-ai/langchain-google** repository), while others are maintained outside of LangChain.