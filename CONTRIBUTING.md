# Engineering House Rules

These guidelines define how we design, build, and maintain our systems. The goal is to maximize predictability, testability, and modularity by treating functional programming not as a syntax, but as a strict discipline of managing state and side effects.

## Core Architectural Principles

### 1. Write Pure Functions by Default
Every function should compute its output solely based on its input parameters. It must not read from global state, modify external variables, or perform hidden operations. If we call a function with the same inputs, it must yield the exact same output every single time.

### 2. Enforce Immutability
Data does not change in place. Once a structure is created, it is strictly read-only. When a state update is necessary, compute and return a new copy of the data structure with the required modifications. This eliminates race conditions and makes the application state completely predictable.

### 3. Push Side Effects to the Edges
The core of the application must consist entirely of pure business logic. Any operation that interacts with the outside world (database reads, network requests, file system access, or logging) is a side effect. These side effects must be isolated and pushed to the outermost boundaries of the system. We build a functional, predictable core wrapped by an imperative shell.

### 4. Treat Errors as Values
Do not use exceptions for control flow. Exceptions are invisible in function signatures and break standard program execution. Instead, represent failures explicitly in your return types. Use constructs like `Option`, `Maybe`, `Either`, or `Result` types. If a function can fail, the compiler must force the caller to handle that failure.

### 5. Separate Data from Execution
Do not bind behavior and data together in heavy objects. Define plain, immutable data structures to represent the state of the system. Define separate modules of pure functions that accept those data structures as inputs and return new data structures as outputs.

### 6. Design for Composition
Small, single-purpose functions are the building blocks of our system. Design functions so their outputs map cleanly to the inputs of other functions. We solve complex problems by chaining together simple, easily testable functions.

## Version Control and Commit Standards

Commit messages act as the permanent log of our engineering decisions. They must be pragmatic, concise, and focused on intent.

### 1. Explain the Why, Not the What
Do not describe the code changes. The diff already tells us what lines were added or removed. The commit message must explain the business or technical reasoning behind the change.

### 2. Keep Commits Atomic
A single commit must represent one logical change. Do not bundle unrelated refactoring, feature additions, and bug fixes into a single massive commit.

### 3. Strict Rule on Metadata
Under no circumstances should commit messages contain URLs to AI chat sessions, generated prompt logs, or personal tracking links. Keep the git history strictly focused on the source code and the engineering context.

### 4. Commit Message Format
Adopt a clean, professional standard format for the first line. Keep it under 50 characters.

Format: `[Scope] Action performed and reasoning`

Examples of acceptable commit messages:

- `fix(auth): prevent race condition on token refresh`
- `perf(db): add composite index to speed up user queries`
- `refactor(api): extract payment validation into pure function`
- `feat(cart): implement immutable cart state calculation`
