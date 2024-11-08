@Library('xmos_jenkins_shared_library@v0.34.0')

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
      defaultValue: '15.3.0',
      description: 'The XTC tools version'
    )
  } // parameters

  environment {
    XMOSDOC_VERSION = "v6.1.2"
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
                  // try building a simple app without venv to check
                  // build that doesn't use design tools won't
                  // need Python
                  withTools(params.TOOLS_VERSION) {
                    dir("test/biquad") {
                      sh "cmake -B build"
                      sh "cmake --build build"
                    } // dir
                  } // tools
                } // dir

                createVenv("lib_audio_dsp/requirements.txt")
                dir("lib_audio_dsp") {
                  // build everything
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      sh "pip install -r requirements.txt"
                      buildApps([
                        "test/biquad",
                        "test/cascaded_biquads",
                        "test/signal_chain",
                        "test/fir",
                        "test/utils",
                      ]) // buildApps
                    } // tools
                  } // withVenv
                } // dir
              } // steps

            } // Build
            stage('Test Biquad') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
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
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
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
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
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
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
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
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
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
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/td_block_fir") {
                          sh "pytest python -m pytest --junitxml='pytest_result.xml' -rA -vvv --durations=0 -o junit_logging=all"
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
                  // try building a simple app without venv to check
                  // build that doesn't use design tools won't
                  // need Python
                  withTools(params.TOOLS_VERSION) {
                    dir("test/biquad") {
                      sh "cmake -B build"
                      sh "cmake --build build"
                    } // dir
                  } // tools
                } // dir
                createVenv("lib_audio_dsp/requirements.txt")
                dir("lib_audio_dsp") {
                  // build everything
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      sh "pip install -r requirements.txt"
                      buildApps([
                        "test/drc",
                        "test/reverb",
                      ]) // buildApps
                    }
                  }
                }
              }
            } // Build
            stage('Test DRC') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      withMounts([["projects", "projects/hydra_audio", "hydra_audio_test_skype"]]) {
                        withEnv(["hydra_audio_PATH=$hydra_audio_test_skype_PATH"]){
                          catchError(stageResult: 'FAILURE', catchInterruptions: false){
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
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/reverb") {
                          runPytest("--dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test Reverb
            stage('Test FD block FIR') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/fd_block_fir") {
                          sh "pytest python -m pytest --junitxml='pytest_result.xml' -rA -vvv --durations=0 -o junit_logging=all"
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
            createVenv("requirements.txt")
            withVenv {
              sh 'pip install -e ./python'
              sh 'pip install "pyright < 2.0"'
              sh 'pip install "ruff < 0.4"'
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
                        sh "python -m pytest -m group0 --junitxml=pytest_result.xml -rA -v --durations=0 -o junit_logging=all --log-cli-level=INFO --adapter-id " + adapterIDs[0]
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
                        sh "python -m pytest -m group1 --junitxml=pytest_result.xml -rA -v --durations=0 -o junit_logging=all --log-cli-level=INFO --adapter-id " + adapterIDs[0]
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
