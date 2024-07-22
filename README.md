<h1 align="center">
    🧨 OptiModel
</h1>
<p align="center">
A smart framework for always calling models in the most efficient and cost effective way possible
</p>
<p align="center">
    <a href="https://lytix.co">
        <img src="https://img.shields.io/badge/Visit%20Us-Lytix-brightgreen" alt="Lytix">
    </a> 👩‍💻 
    <a href="https://badge.fury.io/py/optimodel-server">
        <img src="https://badge.fury.io/py/optimodel-server.svg" alt="PyPI version">
    </a> 💻 
    <a href="https://badge.fury.io/py/optimodel-py">
        <img src="https://badge.fury.io/py/optimodel-py.svg" alt="PyPI version">
    </a>
    <a href="https://discord.gg/8TCbHsSe">
        <img src="https://img.shields.io/badge/Join%20our%20community-Discord-blue" alt="Discord Server">
    </a>
</p>

## [Quickstart](#quickstart)

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

#### Step 3: Add a validator

You can also optionally pass in validators and fallback models to ensure your results are what you expect. Here is an example of a simple JSON validator:

```py
def validator(x) -> bool:
    """
    Simple validator to check if the response is JSON
    """
    try:
        json.loads(x)
        return True
    except:
        return False

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

## [Cloud Quickstart](#cloud-quickstart)

You can also use lytix to host your server for you and interact with it from the cloud. See [here](http://docs.lytix.co/OptiModel/getting-started) for more information.

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

### Groq

**Locally:** Ensure you pass your Groq API key in the following environment variable: `GROQ_API_KEY`

**SAAS Mode:** When running in SAAS mode, the API will expect the credentials param to pass in your Groq key. For example the curl would look like:

```
curl --location 'BASE_URL/query' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [{"role": "user", "content": "hello world!"}],
    "modelToUse": "llama3_8b_instruct",
    "credentials": [{
        "groqApiKey": "<grok-api-key>"
    }]
}'
```

## [Adding A New Provider](#adding-a-new-provider)

You can always add a new provider (for example a custom local model that you'd like to use to save money if possible)

We've tried to make this as easy as possible, There are 3 steps involved:

### Step 1: Build a new provider

We've defined a base provider [here](https://github.com/Lytix-Labs/optimodel/blob/master/server/src/optimodel_server/Providers/BaseProviderClass.py#L15). You'll need to implement the following function:

- `validateProvider(self) => bool`: This is a function that will be used to validate the provider if running in local mode (e.g. do we have the credentials to use this provider)

- `def makeQuery(
    self,
    messages: list[ModelMessage],
    model: ModelTypes,
    temperature: int = 0.2,
    maxGenLen: int = 1024,
    credentials: list[NewModelCredentials] | None = None,
) -> QueryResponse`: This is the function used to make the query. The only field to note is `NewModelCredentials` which is the credentials needed to use the model in SAAS mode.
  - _Note:_ You can ignore `credentials` if you set `supportSAASMode` to `false` in your model definition

### Step 2: Add our new model to the config

You'll need to let our `Config` class know this new provider exists be adding a new `case` statement [here](https://github.com/Lytix-Labs/optimodel/blob/master/server/src/optimodel_server/Config/Config.py#L64)

### Step 3 (optional): Add support for SAAS mode

If you'd like this new provider to be available in SAAS mode, you'll need to add a new `case` statement [here](https://github.com/Lytix-Labs/optimodel/blob/master/server/src/optimodel_server/Planner/Planner.py#L39) (_Note:_ Remember you'll need to implement `credentials` support in your `makeQuery` implementation if you enable this.)

```

```
