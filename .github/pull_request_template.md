# Pull Request

Describe the change. Link the Jira task: {{ JIRA-XXX }}.

## Mandatory Checks

* [ ] Implements exactly what the Jira task specifies
* [ ] Conforms to the code standard and is readable
* [ ] Has tests for happy paths and relevant failures

## Versioning

* Branch name starts with work-item ID: {{ PRD-1234 }}
* Commits tagged `[major]`, `[minor]`, or `[patch]` (default is `[patch]`)

## Rebase Rules

* Keep the branch rebased on `main`
* Do **not** use GitHub’s rebase button

Rebase steps:

* [ ] `git checkout main && git pull`
* [ ] `git checkout {{ feature-branch }}`
* [ ] `git rebase main`
* [ ] Resolve conflicts
* [ ] Run tests
* [ ] `git push --force-with-lease`
