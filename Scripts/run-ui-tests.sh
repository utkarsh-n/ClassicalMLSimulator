#!/bin/bash
# Run native UI tests serially against one explicit Xcode and dedicated simulator.
# Each persistent run keeps its exact staged inputs, commands, phase logs, result
# bundles, and exit statuses. Compatible with the macOS-provided Bash 3.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  DEVELOPER_DIR=/path/to/Xcode.app/Contents/Developer \
    Scripts/run-ui-tests.sh SIMULATOR_UDID [--only-testing TEST_IDENTIFIER]...

The simulator may instead be supplied through SIMULATOR_UDID or --simulator.
Use a simulator reserved for this run. The script waits for it to finish booting,
builds the UI tests, then runs the generated .xctestrun without building again.

Without --only-testing, every ClassicalMLSimulatorUITests test runs serially
(currently 17 tests). Focused filters apply only to the execution phase:
  --only-testing ClassicalMLSimulatorUITests/ClassicalMLSimulatorUITests/testGaussianNaiveBayesEvidenceAndMeans

Runs persist under $HOME/Library/Developer/ClassicalMLSimulatorUIQA/Run.XXXXXX.
UI_TEST_RUN_ROOT may select another absolute persistent directory outside
Documents and temporary storage. Each run contains Source, DerivedData,
Build.xcresult, Results.xcresult, commands, logs, and phase exit statuses.

The script changes neither the global Xcode selection nor debugger authorization,
and does not erase, shut down, or restart simulators.
USAGE
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

simulator_udid="${SIMULATOR_UDID:-}"
simulator_argument_set=false
only_testing=("-only-testing:ClassicalMLSimulatorUITests")
focused_testing=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --simulator)
            [[ $# -ge 2 ]] || fail '--simulator needs a simulator UDID.'
            [[ "$simulator_argument_set" == false ]] || fail 'Supply one simulator UDID.'
            simulator_udid="$2"
            simulator_argument_set=true
            shift 2
            ;;
        --only-testing)
            [[ $# -ge 2 && -n "$2" ]] || fail '--only-testing needs a test identifier.'
            case "$2" in
                ClassicalMLSimulatorUITests|ClassicalMLSimulatorUITests/*)
                    if [[ "$focused_testing" == false ]]; then
                        only_testing=("-only-testing:$2")
                        focused_testing=true
                    else
                        only_testing+=("-only-testing:$2")
                    fi
                    ;;
                *) fail 'Test identifiers must belong to ClassicalMLSimulatorUITests.' ;;
            esac
            shift 2
            ;;
        --*)
            fail "Unknown option: $1"
            ;;
        *)
            [[ "$simulator_argument_set" == false ]] || fail 'Supply one simulator UDID.'
            simulator_udid="$1"
            simulator_argument_set=true
            shift
            ;;
    esac
done

[[ -n "${DEVELOPER_DIR:-}" ]] || fail 'Set DEVELOPER_DIR explicitly to Xcode.app/Contents/Developer.'
[[ "$DEVELOPER_DIR" == /* ]] || fail 'DEVELOPER_DIR must be an absolute path.'
[[ -x "$DEVELOPER_DIR/usr/bin/xcodebuild" ]] || fail "No xcodebuild executable under DEVELOPER_DIR: $DEVELOPER_DIR"
[[ "$simulator_udid" =~ ^[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]] \
    || fail 'Supply a simulator UDID argument, --simulator, or SIMULATOR_UDID.'

script_directory="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_directory="$(cd -P "$script_directory/.." && pwd)"
export DEVELOPER_DIR

build_inputs=(App Sources UITests ClassicalMLSimulator.xcodeproj project.yml Package.swift)
source_inputs=()
for input in "${build_inputs[@]}"; do
    [[ -e "$source_directory/$input" ]] || fail "Missing build input: $input"
    source_inputs+=("$source_directory/$input")
done

run_root="${UI_TEST_RUN_ROOT:-$HOME/Library/Developer/ClassicalMLSimulatorUIQA}"
[[ "$run_root" == /* ]] || fail 'UI_TEST_RUN_ROOT must be an absolute path.'
mkdir -p "$run_root"
run_root="$(cd -P "$run_root" && pwd)"
case "$run_root" in
    "$HOME/Documents"|"$HOME/Documents/"*|/Users/*/Documents|/Users/*/Documents/*|\
    /tmp|/tmp/*|/private/tmp|/private/tmp/*|/var/folders/*|/private/var/folders/*)
        fail 'Use a persistent run directory outside Documents and temporary storage.'
        ;;
esac
run_directory="$(mktemp -d "$run_root/Run.XXXXXX")"
stage_directory="$run_directory/Source"
derived_directory="$run_directory/DerivedData"
build_result_path="$run_directory/Build.xcresult"
test_result_path="$run_directory/Results.xcresult"
mkdir -p "$stage_directory"

# Keep an overall status even when staging, boot readiness, or building fails.
finish() {
    script_status=$?
    trap - EXIT
    printf '%s\n' "$script_status" > "$run_directory/script-exit-status.txt"
    printf '\nRun directory: %s\nScript exit status: %s\n' "$run_directory" "$script_status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Save a replayable command and both the command and log-writer exit statuses.
# PIPESTATUS is captured immediately, preserving a failing Xcode exit through tee.
run_phase() {
    local phase="$1"
    shift
    {
        printf 'DEVELOPER_DIR=%q ' "$DEVELOPER_DIR"
        printf '%q ' "$@"
        printf '\n'
    } > "$run_directory/$phase.command.sh"
    printf '\nPhase: %s\nLog: %s/%s.log\n' "$phase" "$run_directory" "$phase"
    set +e
    "$@" 2>&1 | /usr/bin/tee "$run_directory/$phase.log"
    local phase_status=("${PIPESTATUS[@]}")
    set -e
    local command_status="${phase_status[0]}"
    local log_status="${phase_status[1]}"
    printf '%s\n' "$command_status" > "$run_directory/$phase-exit-status.txt"
    printf '%s\n' "$log_status" > "$run_directory/$phase-log-exit-status.txt"
    printf '%s exit status: %s\n' "$phase" "$command_status"
    if [[ "$command_status" -ne 0 ]]; then
        return "$command_status"
    fi
    if [[ "$log_status" -ne 0 ]]; then
        printf 'Saving the %s log failed.\n' "$phase" >&2
        return "$log_status"
    fi
}

{
    printf 'started_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'source=%s\n' "$source_directory"
    printf 'staged_source=%s\n' "$stage_directory"
    printf 'developer_dir=%s\n' "$DEVELOPER_DIR"
    printf 'simulator_udid=%s\n' "$simulator_udid"
    if [[ "$focused_testing" == false ]]; then
        printf 'scope=full ClassicalMLSimulatorUITests suite\n'
    else
        printf 'scope=focused\n'
        printf 'test_filter=%s\n' "${only_testing[@]}"
    fi
} > "$run_directory/run-info.txt"

printf 'Persistent run directory: %s\n' "$run_directory"
/usr/bin/rsync -a "${source_inputs[@]}" "$stage_directory/"
# -n makes --delete read-only. Any checksum or membership difference means the
# staged build inputs changed during copying, so stop before invoking Xcode.
/usr/bin/rsync -rcln --delete --itemize-changes "${source_inputs[@]}" "$stage_directory/" \
    > "$run_directory/source-parity.txt"
if [[ -s "$run_directory/source-parity.txt" ]]; then
    cat "$run_directory/source-parity.txt" >&2
    fail 'Build inputs changed during staging; inspect source-parity.txt and retry.'
fi

# Only this explicitly supplied simulator is booted. Finish boot readiness before
# beginning either Xcode phase, rather than racing a test launch against boot.
run_phase boot /usr/bin/xcrun simctl bootstatus "$simulator_udid" -b

common_arguments=(
    -destination "platform=iOS Simulator,id=$simulator_udid"
    -destination-timeout 120
    -parallel-testing-enabled NO
    -collect-test-diagnostics never
    -test-timeouts-enabled YES
    -default-test-execution-time-allowance 600
    -maximum-test-execution-time-allowance 900
)
build_command=(
    "$DEVELOPER_DIR/usr/bin/xcodebuild"
    -project "$stage_directory/ClassicalMLSimulator.xcodeproj"
    -scheme ClassicalMLSimulator
    -configuration Debug
    -jobs 2
    -derivedDataPath "$derived_directory"
    -resultBundlePath "$build_result_path"
    "${common_arguments[@]}"
    build-for-testing
)
run_phase build-for-testing "${build_command[@]}"

# Match the file actually emitted by this build, including the SDK/architecture
# suffix chosen by Xcode. Refuse ambiguous or missing products.
xctestrun_path=""
xctestrun_count=0
for candidate in "$derived_directory/Build/Products/"*.xctestrun; do
    [[ -f "$candidate" ]] || continue
    xctestrun_count=$((xctestrun_count + 1))
    xctestrun_path="$candidate"
done
[[ "$xctestrun_count" -eq 1 ]] \
    || fail "Expected exactly one generated .xctestrun; found $xctestrun_count under $derived_directory/Build/Products."
printf 'xctestrun=%s\n' "$xctestrun_path" >> "$run_directory/run-info.txt"

test_command=(
    "$DEVELOPER_DIR/usr/bin/xcodebuild"
    -xctestrun "$xctestrun_path"
    -resultBundlePath "$test_result_path"
    "${common_arguments[@]}"
    "${only_testing[@]}"
    test-without-building
)
printf 'Native test results: %s\n' "$test_result_path"
run_phase test-without-building "${test_command[@]}"
