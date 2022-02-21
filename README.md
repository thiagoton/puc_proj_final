# How to start
First, you need to setup the environment. But before doing that, ensure that you hal a valid copy of the _secret.key_ file. It will be needed during environment setup

```
$ mkdir .local # create a directory called '.local'
$ cp <secret file orig> .local/secret.key # copy provided secret file into .local dir
$ source setenv.sh
```

After it finishes, you will be able to use the repo. Note the setup procedure must be done whenever you start a new session of your terminal.

# Tips
- **Avoid interrupting DVC**:
    It is easy to get into an inconsistent state if user interrupts (CTRL+C, for example) pipeline execution. If you issued a pipeline execution, make sure you can wait if finishes. Otherwise, some erros may occur. Note that you can perform the steps (train, evaluate) by hand!

# Known Issues
## **ERROR: unexpected error - 'cannot stash changes - there is nothing to stash.'**
Sometimes, dvc+git gets into an inconsistent state. If such behavior occurs, a way to workaround it is by making a small modification at a single file that is tracked by git (ex: add a _space_ into this README). This way is known to overcome such problem