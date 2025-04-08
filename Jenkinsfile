@Library('xmos_jenkins_shared_library@v0.35.0')

def boolean hasChangesIn(String module) {
  dir("lib_audio_dsp"){
    if (env.CHANGE_TARGET == null) {
      return false
    } 
    else {
      return sh(
        returnStatus: true,
        script: "git diff --name-only remotes/origin/${env.CHANGE_TARGET}...remotes/origin/${env.BRANCH_NAME} | grep -v .rst | grep ${module}"
      ) == 0
    }
  }
}

def boolean hasGenericChanges() {
    echo env.BRANCH_NAME
    if (env.BRANCH_NAME ==~ /master|main|release.*/) {
      return true
    }
    else if (env.BRANCH_NAME  ==~ /develop/) {
      return true
    }
    else if (hasChangesIn("utils | grep -v reverb_utils")) {
      return true
    }
    else if (hasChangesIn("helpers | grep -v reverb_utils")) {
      return true
    }
    else if (hasChangesIn("adsp")) {
      return true
    }
    else if (hasChangesIn("defines")) {
      return true
    }
    else if (hasChangesIn("generic")) {
      return true
    }
    else {
      return false
    }
}

def runningOn(machine) {
  println "Stage running on:"
  println machine
}

def buildApps(appList) {
  appList.each { app ->
    sh "cmake -G 'Unix Makefiles' -S ${app} -B ${app}/build"
    sh "xmake -C ${app}/build -j\$(nproc)"
  }
}

def versionsPairs = [
    "python/pyproject.toml": /version[\s='\"]*([\d.]+)/,
    "settings.yml": /version[\s:'\"]*([\d.]+)/,
    "CHANGELOG.rst": /(\d+\.\d+\.\d+)/,
    "**/lib_build_info.cmake": /set\(LIB_VERSION \"?([\d.]+)/,
    "README.rst": /:\s*version:\s*([\d.]+)/
]

getApproval()
pipeline {
  agent none

  parameters {
    string(
      name: 'TOOLS_VERSION',
      defaultValue: '15.3.1',
      description: 'The XTC tools version'
    )
  } // parameters

  environment {
    XMOSDOC_VERSION = "v6.3.1"
    HAS_GENERIC_CHANGES = false
  } // environment

  options {
    skipDefaultCheckout()
    timestamps()
    buildDiscarder(xmosDiscardBuildSettings(onlyArtifacts=false))
  } // options

  stages {
    stage('Stop previous builds') {
      when {
        // don't stop runs on develop or main
        not {
          anyOf {
            branch "main"
            branch "develop"
          }
        }
      }
      steps {
        stopPreviousBuilds()
      }
    } // Stop previous builds
    stage('CI') {
      parallel {
        stage ('Build & Test') {
          agent {
            label 'linux&&x86_64'
          }
          stages {
            stage ('Build') {
              steps {
                runningOn(env.NODE_NAME)
                dir("lib_audio_dsp") {
                  checkout scm
                  script{
                    env.HAS_GENERIC_CHANGES = hasGenericChanges().toBoolean()
                  }
                  echo "env.HAS_GENERIC_CHANGES is '${env.HAS_GENERIC_CHANGES}'"
                  echo "env.HAS_GENERIC_CHANGES is '${hasGenericChanges()}'"
                  // try building a simple app without venv to check
                  // build that doesn't use design tools won't
                  // need Python
                  withTools(params.TOOLS_VERSION) {
                    dir("test/biquad") {
                      sh "cmake -B build"
                      sh "cmake --build build -j\$(nproc)"
                    } // dir
                  } // tools
                } // dir
                createVenv("lib_audio_dsp/requirements.txt")
                dir("lib_audio_dsp") {
                  // build everything
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      sh "pip install -r requirements.txt"
                    } // tools
                  } // withVenv
                } // dir
              } // steps

            } // Build
            stage('Test Biquad') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("biquad")}
                  }
              }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        buildApps(["test/biquad"])
                        dir("test/biquad") {
                          runPytest("--dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test biquad
            stage('Test Cascaded Biquads') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("biquad")}
                  expression{hasChangesIn("cascaded_biquad")}
                  }
                }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        buildApps(["test/cascaded_biquads"])
                        dir("test/cascaded_biquads") {
                          runPytest("--dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test cascaded biquad
            stage('Unit tests') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/unit_tests") {
                          runPytest("--dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // Unit tests
            stage('Test Utils') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("utils")}
                  expression{hasChangesIn("control")}
                  }
                }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        buildApps(["test/utils"])
                        dir("test/utils") {
                          runPytest("--dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test utils
            stage('Test FIR') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("fir")}
                  }
                }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        buildApps(["test/fir"])
                        dir("test/fir") {
                          runPytest("--dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test SC
            stage('Test SC') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("signal_chain")}
                  }
                }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        buildApps(["test/signal_chain"])
                        dir("test/signal_chain") {
                          runPytest("--dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test SC
            stage('Test TD block FIR') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("td_block")}
                  }
                }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/td_block_fir") {
                          runPytest("--dist worksteal --durations=0")
                        }
                      }
                    }
                  }
                }
              }
            } // test TD block FIR
          }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Build and test
        stage ('Build & Test 2') {
          agent {
            label 'linux&&x86_64'
          }
          stages {
            stage ('Build') {
              steps {
                runningOn(env.NODE_NAME)
                dir("lib_audio_dsp") {
                  checkout scm
                  script{
                    env.HAS_GENERIC_CHANGES = hasGenericChanges()
                  }
                  // try building a simple app without venv to check
                  // build that doesn't use design tools won't
                  // need Python
                  withTools(params.TOOLS_VERSION) {
                    dir("test/biquad") {
                      sh "cmake -B build"
                      sh "cmake --build build -j\$(nproc)"
                    } // dir
                  } // tools
                } // dir
                createVenv("lib_audio_dsp/requirements.txt")
                dir("lib_audio_dsp") {
                  // build everything
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      sh "pip install -r requirements.txt"
                    }
                  }
                }
              }
            } // Build
            stage('Test DRC') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("-e drc -e env -e limit -e noise -e compressor")}
                  }
                }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      withMounts([["projects", "projects/hydra_audio", "hydra_audio_test_skype"]]) {
                        withEnv(["hydra_audio_PATH=$hydra_audio_test_skype_PATH"]){
                          catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        buildApps(["test/drc"])
                            dir("test/drc") {
                              runPytest("--dist worksteal")
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            } // test drc
            stage('Test Reverb') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("reverb")}
                  }
                }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        buildApps(["test/reverb"])
                        dir("test/reverb") {
                          runPytest("--dist worksteal --durations=0")
                        }
                      }
                    }
                  }
                }
              }
            } // test Reverb
            stage('Test FD block FIR') {
              when {
                anyOf {
                  expression{hasGenericChanges()}
                  expression{hasChangesIn("fd_block")}
                  }
                }
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/fd_block_fir") {
                          runPytest("--dist worksteal --durations=0")
                        }
                      }
                    }
                  }
                }
              }
            } // test FD block FIR
          }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Build and test 2

        stage('Style and docs') {

          agent {
            label 'documentation&&linux&&x86_64'
          }
          steps {
            checkout scm
            createVenv("requirements-format.txt")
            withVenv {
              sh 'pip install --no-deps -r requirements-format.txt'
              sh "make -C python check" // ruff check
              versionChecks checkReleased: false, versionsPairs: versionsPairs
              buildDocs xmosdocVenvPath: "${WORKSPACE}", archiveZipOnly: true // needs python run
            }
          }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Style and docs

        stage ('Hardware Test') {
          agent {
            label 'xcore.ai && uhubctl'
          }

          steps {
            runningOn(env.NODE_NAME)
            sh 'git clone https://github0.xmos.com/xmos-int/xtagctl.git'
            dir("lib_audio_dsp") {
              checkout scm
            }
            createVenv("lib_audio_dsp/requirements.txt")

            dir("lib_audio_dsp") {
              withVenv {
                withTools(params.TOOLS_VERSION) {
                  sh "pip install -r requirements.txt"
                  sh "pip install -e ${WORKSPACE}/xtagctl"
                  withXTAG(["XCORE-AI-EXPLORER"]) { adapterIDs ->
                      sh "xtagctl reset ${adapterIDs[0]}"
                      dir("test/pipeline") {
                        sh "python -m pytest -m group0 -n auto --junitxml=pytest_result.xml -rA -v --durations=0 -o junit_logging=all --log-cli-level=INFO --adapter-id " + adapterIDs[0]
                      }
                  }
                }
              }
            }
          }

          post {
            cleanup {
              xcoreCleanSandbox()
            }
            always {
              dir("${WORKSPACE}/lib_audio_dsp/test/pipeline") {
                junit "pytest_result.xml"
              }
            }
          }
        } // Hardware test

        stage ('Hardware Test 2') {
          agent {
            label 'xcore.ai && uhubctl'
          }

          steps {
            runningOn(env.NODE_NAME)
            sh 'git clone https://github0.xmos.com/xmos-int/xtagctl.git'
            dir("lib_audio_dsp") {
              checkout scm
            }
            createVenv("lib_audio_dsp/requirements.txt")

            dir("lib_audio_dsp") {
              withVenv {
                withTools(params.TOOLS_VERSION) {
                  sh "pip install -r requirements.txt"
                  sh "pip install -e ${WORKSPACE}/xtagctl"
                  withXTAG(["XCORE-AI-EXPLORER"]) { adapterIDs ->
                      sh "xtagctl reset ${adapterIDs[0]}"
                      dir("test/pipeline") {
                      sh "python -m pytest   -m group1 -n auto --junitxml=pytest_result.xml -rA -v --durations=0 -o junit_logging=all --log-cli-level=INFO "
                    }
                  }
                }
              }
            }
          }

          post {
            cleanup {
              xcoreCleanSandbox()
            }
            always {
              dir("${WORKSPACE}/lib_audio_dsp/test/pipeline") {
                junit "pytest_result.xml"
              }
            }
          }
        } // Hardware test 2
      } // parallel
    } // CI
  } // stages
} // pipeline
