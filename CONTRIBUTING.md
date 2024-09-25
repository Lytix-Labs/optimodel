# Contributing to Optimodel 🧱

We have a couple of guides to quickly get started adding a new provider or guard.

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
