# **GRB**

GRB stands for Gamma-Ray Burst.

* [Directories](https://github.com/Perun21/GRB#directories)
* [Requirements](https://github.com/Perun21/GRB#requirements)
* [Repository Managment](https://github.com/Perun21/GRB#repository-managment)
  * [Commit message convention](https://github.com/Perun21/GRB#type-must-be-one-of-the-following-mentioned-below)

## **Directories**

* **Docs** : The directory in which all the documents relating to the project exist.

    It currently contains:
  * The conical morphology, the jetted GeV emission, the-X ray afterglows of Long GRBs
    * ukwatta2016
    * wang
    * X-ray Flares

* **Regression for Redshifts** : This directory contains jupyter notebooks regarding calculating redshifts using regression method.

* **Utils** : The directory in which all code based utilities and tools exist.

    It currently contains:
  * **txt2csv** : This python module converts a text file to csv.
  * **lc python file**
    * **Random Forest for SDSS** : Random Forest regression algorithm for estimating photometric redshift of galaxies.

## **Requirements**

For each new package you add to the project, add the name of the package to the `requirements.in` file and use the command below and the `requirements.txt` file will be updated:

```pip-compile requirements.in```

## **Repository Managment**

* Code

  * **NEVER** make your changes on the main branch.
  * **ALWAYS**  create a new branch and make your changes on that.

  * Push your branch and make a new pull request.
  * Pull requests will be merged after reviewing.

* Images and Document files
  * There is no need to create new branches for pushing these kind of files.

  * If you use the upload method on github website, remember to change the defualt commit message.

* Commit Message

    Follow the git message convention below:

    ```text
    <type>[scope]: <description>

    [optional body]
    ```

### **type** must be one of the following mentioned below

* `build` : Build related changes (eg: adding external dependencies)
* `chore` : A code change that external user won't see (eg: change to .gitignore file)
* `feat` : A new feature
* `fix` : A bug fix
* `docs` : Documentation related changes
* `refactor` :  A code that neither fix bug nor adds a feature. (eg: You can use this when there is semantic changes like renaming a variable/ function name)
* `perf` : A code that improves performance
* `style` : A code that is related to styling
* `test` : Adding new test or making changes to existing test

### **scope** is optional

* Scope must be noun and it represents the section of the section of the codebase

### **description**

* use imperative, present tense (eg: use "add" instead of "added" or "adds")
* don't use dot(.) at end
* don't capitalize first letter

    visit this [link](https://www.conventionalcommits.org/en/v1.0.0/) for more info.
