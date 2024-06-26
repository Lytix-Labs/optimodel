<h1 align="center">
    Optimodel
</h1>
<p align="center">
A smart framework for always calling models in the most efficient and cost effective way possible
</p>

**Server** [![PyPI version](https://badge.fury.io/py/optimodel-server.svg)](https://badge.fury.io/py/optimodel-server) | **SDK** [![PyPI version](https://badge.fury.io/py/optimodel-py.svg)](https://badge.fury.io/py/optimodel-py)

### Quickstart

To get started you'll need to setup the server and then use the SDK to call the models.

#### Prerequisites

First decide what providers you'd like to setup. Please see [Supported Providers](#supported-providers) below to setup the providers you'd like to use and the pre-requisites for each provider.

#### Step 1: Setup the server

Now we can install our server and get it running in the background

```sh
$ pip3 install optimodel-server
$ optimodel-server
```

#### Step 2: Call our server

Now we can call our server from our python code. See the example block on what the code might look like. ,

```py
async def main():
    prompt = "Hello How are you?"

    response = await queryModel(
        model=ModelTypes.llama_3_8b_instruct,
        messages=[
            ModelMessage(
                role="system",
                content="You are a helpful assistant. Always respond in JSON syntax",
            ),
            ModelMessage(role="user", content=prompt),
        ],
        speedPriority="low",
        validator=validator,
        fallbackModels=[ModelTypes.llama_3_8b_instruct],
        maxGenLen=256,
    )
    print("Got response:", response)


if __name__ == "__main__":
    asyncio.run(main())
```

Just make sure to setup our `OPTIMODEL_BASE_URL` envvar correctly:

```sh
$ OPTIMODEL_BASE_URL="http://localhost:8000/optimodel/api/v1/" python3 example.py
```

## [Supported Providers](#supported-providers)

### AWS Bedrock

**Locally:** Ensure your current IAM user (`aws sts get-caller-identity`) has the correct permissions needed to call bedrock.

**SAAS Mode:** When running in SAAS mode, the API will expect the credentials param to pass in your AWS accessKey, secretKey and region. For example the curl would look like:

```
curl --location 'BASE_URL/query' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [{"role": "user", "content": "hello world!"}],
    "modelToUse": "llama3_8b_instruct",
    "credentials": [{
        "awsAccessKeyId": "<access-key-id>",
        "awsSecretKey": "<secret-access-key>",
        "awsRegion": "<region>"
    }]
}'
```

### OpenAI

**Locally:** Ensure you pass your OpenAI key in the following environment variable: `OPEN_AI_KEY`

**SAAS Mode:** When running in SAAS mode, the API will expect the credentials param to pass in your openAi key. For example the curl would look like:

```
curl --location 'BASE_URL/query' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [{"role": "user", "content": "hello world!"}],
    "modelToUse": "llama3_8b_instruct",
    "credentials": [{
        "openAiKey": "<openai-key>"
    }]
}'
```

### Together AI

**Locally:** Ensure you pass your Together AI key in the following environment variable: `TOGETHER_API_KEY`

**SAAS Mode:** When running in SAAS mode, the API will expect the credentials param to pass in your Together AI key. For example the curl would look like:

```
curl --location 'BASE_URL/query' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [{"role": "user", "content": "hello world!"}],
    "modelToUse": "llama3_8b_instruct",
    "credentials": [{
        "togetherApiKey": "<together-api-key>"
    }]
}'
```
