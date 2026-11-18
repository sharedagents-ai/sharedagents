# Amazon Bedrock (native Converse)

sharedagents talks to Bedrock through the **native Converse / ConverseStream API**
(not the OpenAI-compatible endpoint). Install the optional SDK:

```bash
pip install 'sharedagents[bedrock]'          # release
pip install -e '.[bedrock]'             # editable clone
```

With AWS credentials already in the service environment, native agents discover
Bedrock models automatically; no provider entry or setup wizard is required.
Discovery uses `AWS_REGION`, then `AWS_DEFAULT_REGION`, then the AWS profile's
region, and defaults to **`sa-east-1`** when none is set. Environment IAM keys
(including temporary session tokens), Bedrock bearer tokens, explicit AWS
profiles, shared credentials, and container/web-identity credentials are supported.
It does not probe EC2 instance metadata just to populate the picker.

At model load, sharedagents lists active streaming text foundation models and inference
profiles. Results are cached for five minutes within each process; the model
picker's refresh bypasses that cache. Catalogue reads make no inference calls.
Removing credentials removes automatic rows on the next config reload. Other
engines and the selected default model are unchanged. AWS model access and
Converse/tool support still depend on the model and account.

For custom aliases, credentials or context limits, add a model row with
`api: "bedrock-converse"` (the explicit marker that
selects the transport) or run `sharedagents configure` and choose
`b. Amazon Bedrock`. Example row (illustrative — **no credentials**):

```json
{
  "alias": "bedrock-primary",
  "api": "bedrock-converse",
  "backend": "bedrock",
  "provider": "bedrock",
  "baseUrl": "bedrock://us-east-1",
  "region": "us-east-1",
  "id": "<operators-supply-a-real-foundation-id-or-inference-profile-arn>",
  "auth": "aws-iam",
  "contextWindow": 200000,
  "maxTokens": 8192
}
```

- Manual rows take precedence over discovered rows for the same region/model.
  AWS catalogue responses omit context limits, so automatic rows use a conservative
  4,096-token context budget and 1,024-token output limit. Set a manual row to use
  the model's verified larger limits. Discovery never saves credentials in rows.
- `baseUrl` is a `bedrock://REGION` display URL only; the SDK derives the real
  HTTPS endpoint from `region`. A `bedrock://` URL is never parsed as HTTP.

### Authentication

Both modes are supported; there is **no silent fallback** between them.

- **`auth: "aws-iam"`** (recommended): the normal boto3 chain — IAM role,
  instance/container credentials, shared config or `AWS_PROFILE` (`profile` on
  the row overrides `AWS_PROFILE` in that child worker only). AWS recommends
  short-lived role credentials over long-term keys.
- **`auth: "bedrock-api-key"`**: a Bedrock bearer token, resolved per-agent
  first (`secrets.env` / `api_keys`) then inherited environment, under
  `apiKeyEnv` (default `AWS_BEARER_TOKEN_BEDROCK`). An explicit empty value
  means "no key" and blocks the environment fallback. The value is set only in
  the child worker's environment at request time and is **never** written to
  `config.json`, sessions, logs or the request JSON.

### Permissions and pricing

- Least privilege: `bedrock:InvokeModelWithResponseStream` for normal turns,
  plus `bedrock:InvokeModel` if a one-shot path (vision transcription,
  compaction, distillation) is used.
- Automatic discovery also needs `bedrock:ListFoundationModels` and
  `bedrock:ListInferenceProfiles`. A profile-list denial still allows direct
  foundation models; discovery failure leaves manual rows available.
- Enable model access and any cross-region inference profile in the AWS
  console for the target account/region.
- Costs come from the metrics the worker reports (provider `bedrock`) plus the
  operator's `~/.sharedagents/pricing.json`. Bedrock model/region prices vary; set
  exact model or alias overrides from the AWS price sheet rather than a blanket
  rate.

### Compatibility and limitations

- Streaming text, client-side tool calls/results and image blocks are
  supported where the chosen model supports them.
- **Extended reasoning is off in v1**: no `additionalModelRequestFields`
  thinking, the picker/`/effort` marks depth unsupported, and signed
  `reasoningContent` during a tool turn fails closed rather than replaying an
  unsigned assistant turn.
- `sharedagents doctor --health` checks a Bedrock route **offline** (region/auth and
  credential presence). `sharedagents doctor --bedrock-smoke` makes an explicit,
  opt-in, billable live call; it is never run by `--health`.
