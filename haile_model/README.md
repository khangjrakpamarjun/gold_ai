# Models

Repository that conains code to train models and produce recommendations

## Important

### Test

Before committing any changes, please make sure following commands run without errors,
as they are used in jobs to run pipelines in cluster:

```sh
kedro run --pipeline live_upstream_pipe --env live
kedro run --pipeline live_full_pipe --env live
```

### CI Merge Requirements

All merge requests into the default branch (i.e Develop) must pass the following CI merge requirements:

1. [Linting] [Black code formatter](https://black.readthedocs.io/en/stable/)
2. [Linting] [isort import formatter](https://pycqa.github.io/isort/)
3. [Testing] [Kedro Pipeline Test](https://gitlab.tools.simonai.draslovka.com/tenants/simonai/docs/-/wikis/Testing-DS-Pipeline):

   a. **_Requires_** CURRENT_MODEL_VERSION variable to be upto date in project group's [CICD variable](https://gitlab.tools.simonai.draslovka.com/groups/tenants/oceana/-/settings/ci_cd#Variables), with the latest model release version.

   b. To update Model Data (e.g. retraining, updated test/control features,etc.): Create a new image using [Image Build steps](https://gitlab.tools.simonai.draslovka.com/tenants/oceana/haile_model#image-build). Update the CURRENT_MODEL_VERSION variable (higlighed above) to the new tag.

Tests are auto started on a MR to the default branch and can be viewed [here](https://gitlab.tools.simonai.draslovka.com/tenants/oceana/haile_model/-/pipelines)

> **Run the following shell script (from root directory) to do the linting for you (prior to creating a MR) `./lint.sh`**

## Image build

To trigger the image build, execute `./build.sh` script in the project's root.

The `./build.sh` script does following:

1. Increments project version in pyproject.toml
2. Creates a tag with the new version from the step above
3. Pushes the tag to the repository

Each tag push to the repository triggers build process defined in `./gitlab-ci.yml`.

## Get started using devspace for local development

Local dev environment with Kubernetes is enabled using [devspace.sh](https://www.devspace.sh/docs/getting-started/introduction)

1. Set up local environment, as described [here](https://gitlab.tools.simonai.draslovka.com/tenants/simonai/docs/-/wikis/Remote-Development)
2. Run `./set-devspace.sh YOUR_NAME`
3. Start the dev environment with `devspace dev`
4. when exiting, to log back into devspace use `devspace attach` and select your pod
5. In case of full pod restart run `devspace purge` and then `devspace dev`

## Useful devspace commands

- `devspace COMMAND --help`
- `devspace dev -d` will redeploy, useful when you have changed values
- `devspace logs --stream` to stream logs of a specific pod only
- `devspace purge` to clean all caches and rebuild (should rarely be needed)
