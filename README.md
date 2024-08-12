<h1 align="center">
    🧨 OptiModel
</h1>
<p align="center">
Guards and protection agnostic to your model or provider
</p>
<p align="center">
    <a href="https://lytix.co">
        <img src="https://img.shields.io/badge/Visit%20Us-Lytix-brightgreen" alt="Lytix">
    </a>👩‍💻 
    <a href="https://discord.gg/8TCbHsSe">
        <img src="https://img.shields.io/badge/Join%20our%20community-Discord-blue" alt="Discord Server">
    </a>
</p>

#### Packages

[Server](https://badge.fury.io/py/optimodel-server) <img src="https://badge.fury.io/py/optimodel-server.svg" alt="PyPI version"> |
[Python Client](https://badge.fury.io/py/optimodel-py) <img src="https://badge.fury.io/py/optimodel-py.svg" alt="PyPI version"> |
[Node.js Client](https://badge.fury.io/js/@lytix%2Fclient) <img src="https://badge.fury.io/js/@lytix%2Fclient.svg" alt="npm version"> | [Guard Server](https://badge.fury.io/py/optimodel-guard-server.svg) <img src="https://badge.fury.io/py/optimodel-guard-server.svg" alt="PyPI version">

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

## [Architecture Diagram](#architecture-diagram)

<p align="center">
    <img src="./assets/optimodel-diagram.jpg" alt="OptiModel Architecture Diagram">
</p>

## [Guards](#guards)

```py
response = await queryModel(
    ....
    messages=[
        ModelMessage(
            role="system",
            content="You are a helpful assistant. Always respond in JSON syntax",
        ),
        ModelMessage(role="user", content=prompt),
    ],
    guards=[...] # Optional guards param
)
```

You can also use Lytix to **prevent or alert** when certain types of guards are active. See [here](#adding-a-new-guard) on how to add a guard of your own.

### Blocking requests

In addition to the guard itself, you can also use the `blockRequest` flag to block requests when a guard is active and give a custom message to return instead.

```py
guards=[MicrosoftPresidioConfig(
    guardName="MICROSOFT_PRESIDIO_GUARD",
    guardType="preQuery",
    entitiesToCheck=["EMAIL_ADDRESS"],
    blockRequest=True, # Pass this to block the request
    blockRequestMessage="You are not allowed to ask about this email address" # Pass this to give a custom message
)]
```

### meta-llama/Prompt-Guard-86M <img src="./assets/logos/meta-logo-small.png" alt="Lytix" height=18>

Utilize Meta's prompt guard to protect against jailbreaks and injection attacks. See the model card [here](https://huggingface.co/meta-llama/Prompt-Guard-86M) for more information.

_Note: We recommend starting with only jailbreak with a value of 0.999 unless you know what you are doing_

```py
guards=[LLamaPromptGuardConfig(
    guardName="LLamaPromptGuard",
    jailbreakThreshold=0.9999,
    guardType="preQuery", # You'll likely only want to guard the input here
)]
```

### microsoft/Presidio-Guard <img src="./assets/logos/microsoft-logo-small.png" alt="Lytix" height=18>

Utilize Microsoft's Presidio Guard to protect against PII. See the model card [here](https://microsoft.github.io/presidio/) for more information.

```py
guards=[MicrosoftPresidioConfig(
    guardName="MICROSOFT_PRESIDIO_GUARD",
    guardType="preQuery",
    entitiesToCheck=["EMAIL_ADDRESS"], # See the model card for the full list of entities to check
)]
```

### lytix/Regex-Guard <img src="./assets/logos/lytix-logo.svg" alt="Lytix" height=18>

Simple regex guard to protect against given regex patterns. See [here](https://github.com/Lytix-Labs/optimodel/blob/master/guardServer/src/optimodel_guard/Guards/RegexGuard.py#L54) for source code on how its implemented.

```py
guards=[LytixRegexConfig(
    guardName="LYTIX_REGEX_GUARD",
    regex="secrets",
    guardType="preQuery",
)]
```

## [Supported Providers](#supported-providers)

### AWS Bedrock <img src="./assets/logos/aws-bedrock-logo-small.png" alt="AWS Bedrock" height=18>

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

### OpenAI <img src="./assets/logos/open-ai-logo-small.png" alt="OpenAI" height=18>

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

### Together AI <img src="./assets/logos/together-ai-logo-small.png" alt="Together AI" height=18>

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

### Groq <img src="./assets/logos/groq-logo-small.png" alt="Groq" height=18>

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

### Anthropic <img src="./assets/logos/anthropic-logo-small.png" alt="Anthropic" height=18>

**Locally:** Ensure you pass your Anthropic API key in the following environment variable: `ANTHROPIC_API_KEY`

**SAAS Mode:** When running in SAAS mode, the API will expect the credentials param to pass in your Anthropic key. For example the curl would look like:

```
curl --location 'BASE_URL/query' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [{"role": "user", "content": "hello world!"}],
    "modelToUse": "llama3_8b_instruct",
    "credentials": [{
        "anthropicApiKey": "<anthropic-api-key>"
    }]
}'
```

## [Adding A New Provider](#adding-a-new-provider)

You can always add a new provider (for example a custom local model that you'd like to use to save money if possible)

We've tried to make this as easy as possible, There are 3 steps involved:

### Step 1: Build a new provider

We've defined a base provider [here](https://github.com/Lytix-Labs/optimodel/blob/master/server/src/optimodel_server/Providers/BaseProviderClass.py#L15). You'll need to implement the following function:

- `validateProvider(self) => bool`: This is a function that will be used to validate the provider if running in local mode (e.g. do we have the credentials to use this provider)

- `makeQuery(
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

### Step 3: Add it to our providers enum

You'll need to add your new provider to the `Providers` enum [here](https://github.com/Lytix-Labs/optimodel/blob/master/server/src/optimodel_server_types/__init__.py#L31)

### Step 4 (optional): Add support for SAAS mode

If you'd like this new provider to be available in SAAS mode, you'll need to add a new `case` statement [here](https://github.com/Lytix-Labs/optimodel/blob/master/server/src/optimodel_server/Planner/Planner.py#L39) (_Note:_ Remember you'll need to implement `credentials` support in your `makeQuery` implementation if you enable this.)

## [Adding A New Guard](#adding-a-new-guard)

You can also use Lytix to prevent or alert when certain types of guards are active.

### Step 1: Build a new guard

We've defined a base class that needs to be implemented [here](https://github.com/Lytix-Labs/optimodel/blob/master/guardServer/src/optimodel_guard/Guards/GuardBaseClass.py#L5). You'll need to implement the following function:

- `handlePreQuery(self, query: QueryParams) -> bool`: This function will be invoked if you pass in `guardType: "preQuery"` into the guard config. Implementation of this should only target `user` or `system` messages, not responses (e.g. `assistant` messages). This will be called **before** the model has been called.

- `handlePostQuery(self, query: QueryParams, response: QueryResponse) -> bool:`: This function will be invoked if you pass in `guardType: "postQuery"` into the guard config. Implementation of this should only target `assistant` messages, not `user` or `system` messages. This will be called **after** the model has been called and the response has been generated.

Define a new interface for your guards config [here](https://github.com/Lytix-Labs/optimodel/blob/master/server/src/optimodel_server_types/__init__.py#L121) if you want to pass in custom fields to your guard.

```py
class NewTypeOfGuard(GuardQueryBase):
    guardName: Literal["NEW_TYPE_OF_GUARD"]
    # Add any additional fields here
    foo: str
```

### Step 2: Add support in `optimodel-guard-server`

Update [this](https://github.com/Lytix-Labs/optimodel/blob/master/guardServer/src/optimodel_guard/Guards/__init__.py#L5) file to include the new guard you have created

### Step 3: Add support in the `optimodel-server`

We need to update our types to support passing in our new guard type. Update this type [here](https://github.com/Lytix-Labs/optimodel/blob/master/server/src/optimodel_server_types/__init__.py#L126) and create a new type that you can use

```py
Guards = LLamaPromptGuardConfig | LytixRegexConfig | MicrosoftPresidioConfig | NewTypeOfGuard
```

All done 🚀, you should now be able to call your new guard when querying:

```py
response = await queryModel(
    ....
    messages=[
        ModelMessage(
            role="system",
            content="You are a helpful assistant. Always respond in JSON syntax",
        ),
        ModelMessage(role="user", content=prompt),
    ],
    guards=[NewTypeOfGuard(
        guardName="NEW_TYPE_OF_GUARD",
        # Add any additional fields here
        foo="bar",
    )]
)
```
