@Library('xmos_jenkins_shared_library@v0.32.0')

def runningOn(machine) {
  println "Stage running on:"
  println machine
}

getApproval()
pipeline {
  agent none

  parameters {
    string(
      name: 'TOOLS_VERSION',
      defaultValue: '15.2.1',
      description: 'The XTC tools version'
    )
    string(
      name: 'XCOMMON_CMAKE_VERSION',
      defaultValue: 'v1.0.0',
      description: 'The xcommon cmake version'
    )
  } // parameters

  environment {
    XMOSDOC_VERSION = "v5.2.0"
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
                sh "git clone -b ${params.XCOMMON_CMAKE_VERSION} git@github.com:xmos/xcommon_cmake"
                sh 'git -C xcommon_cmake rev-parse HEAD'
                dir("lib_audio_dsp") {
                  checkout scm
                  // try building a simple app without venv to check
                  // build that doesn't use design tools won't
                  // need python
                  withTools(params.TOOLS_VERSION) {
                    withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                      dir("test/biquad") {
                        sh "cmake -B build"
                        sh "cmake --build build"
                      }
                    }
                  }

                }
                createVenv("lib_audio_dsp/requirements.txt")
                dir("lib_audio_dsp") {
                  // build everything
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      sh "pip install -r requirements.txt"
                      withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                        script {
                          [
                          "test/biquad",
                          "test/cascaded_biquads",
                          "test/drc",
                          ].each {
                            sh "cmake -S ${it} -B ${it}/build"
                            sh "xmake -C ${it}/build -j"
                          }
                        }
                      }
                    }
                  }
                }
              }
            } // Build
            stage('Test Biquad') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/biquad") {
                          runPytest("test_biquad_python.py --dist worksteal")
                          runPytest("test_biquad_c.py --dist worksteal")
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
                          runPytest("test_cascaded_biquads_python.py --dist worksteal")
                          runPytest("test_cascaded_biquads_c.py --dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test cascaded biquad
            stage('Test DRC') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      withMounts([["projects", "projects/hydra_audio", "hydra_audio_test_skype"]]) {
                        withEnv(["hydra_audio_PATH=$hydra_audio_test_skype_PATH"]){
                          catchError(stageResult: 'FAILURE', catchInterruptions: false){
                            dir("test/drc") {
                              runPytest("test_drc_python.py --dist worksteal")
                              runPytest("*_c.py --dist worksteal")
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            } // test drc
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
                sh "git clone -b ${params.XCOMMON_CMAKE_VERSION} git@github.com:xmos/xcommon_cmake"
                sh 'git -C xcommon_cmake rev-parse HEAD'
                dir("lib_audio_dsp") {
                  checkout scm
                  // try building a simple app without venv to check
                  // build that doesn't use design tools won't
                  // need python
                  withTools(params.TOOLS_VERSION) {
                    withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                      dir("test/biquad") {
                        sh "cmake -B build"
                        sh "cmake --build build"
                      }
                    }
                  }

                }
                createVenv("lib_audio_dsp/requirements.txt")
                dir("lib_audio_dsp") {
                  // build everything
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      sh "pip install -r requirements.txt"
                      withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                        script {
                          [
                          "test/reverb",
                          "test/signal_chain",
                          "test/fir"
                          ].each {
                            sh "cmake -S ${it} -B ${it}/build"
                            sh "xmake -C ${it}/build -j"
                          }
                        }
                      }
                    }
                  }
                }
              }
            } // Build
            stage('Unit tests') {
              steps {
                dir("lib_audio_dsp") {
                  withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
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
              }
            } // Unit tests
            stage('Test FIR') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/fir") {
                          runPytest("test_fir_python.py --dist worksteal")
                          runPytest("test_fir_c.py --dist worksteal")
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
                          runPytest("test_signal_chain_python.py --dist worksteal")
                          runPytest("test_signal_chain_c.py --dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test SC
            stage('Test Reverb') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      catchError(stageResult: 'FAILURE', catchInterruptions: false){
                        dir("test/reverb") {
                          runPytest("test_reverb_python.py --dist worksteal")
                          runPytest("test_reverb_c.py --dist worksteal")
                        }
                      }
                    }
                  }
                }
              }
            } // test Reverb
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
          }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Build and test 2

        stage('Style and package') {
          agent {
            label 'linux&&x86_64'
          }
            steps {
              dir("lib_audio_dsp") {
                checkout scm
              }
              createVenv("lib_audio_dsp/requirements.txt")
              dir("lib_audio_dsp") {
                withVenv {
                  withTools(params.TOOLS_VERSION) {
                    dir("python") {
                      sh "pip install ."
                      sh 'pip install "pyright < 2.0"'
                      sh 'pip install "ruff < 0.4"'
                      sh "make check"
                    }
                  }
                }
              }
            }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Style and package

        stage('docs') {

          agent {
            label 'documentation'
          }
          steps {
            checkout scm
            createVenv("requirements.txt")
            withVenv {
              sh 'pip install -e ./python'
              sh "pip install git+ssh://git@github.com/xmos/xmosdoc@${XMOSDOC_VERSION}"
              sh 'xmosdoc'
            }

            archiveArtifacts artifacts: "doc/_out/**"
            zip zipFile: "lib_audio_dsp_docs.zip", archive: true, dir: "doc/_out"
          }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // docs

        stage ('Hardware Test') {
          agent {
            label 'xcore.ai && uhubctl'
          }

          steps {
            runningOn(env.NODE_NAME)
            sh "git clone -b ${params.XCOMMON_CMAKE_VERSION} git@github.com:xmos/xcommon_cmake"
            sh 'git -C xcommon_cmake rev-parse HEAD'
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
                  withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                    withXTAG(["XCORE-AI-EXPLORER"]) { adapterIDs ->
                      sh "xtagctl reset ${adapterIDs[0]}"
                      dir("test/pipeline") {
                        sh "python -m pytest --junitxml=pytest_result.xml -rA -v --durations=0 -o junit_logging=all --log-cli-level=INFO --adapter-id " + adapterIDs[0]
                      }
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

        // Host app on Windows
        stage ('Win32 Host Build & Test') {
          agent {
            label 'sw-bld-win0'
          }
          stages {
            stage ('Build') {
              steps {
                runningOn(env.NODE_NAME)
                // build
                dir("lib_audio_dsp") {
                  checkout scm
                }
                dir('lib_audio_dsp/host') {
                  // Enable the XTC tools for xSCOPE
                  withTools(params.TOOLS_VERSION) {
                    withVS('vcvars32.bat') {
                      bat 'cmake -G "Ninja" -B build -DTESTING=ON'
                      bat 'cd build && ninja'
                    }
                  }
                }
              }
            }
            stage ('Test') {
              steps {
                dir("lib_audio_dsp") {
                  createVenv("requirements.txt")
                  withVenv{
                    // Enable the XTC tools for xSCOPE
                    withTools(params.TOOLS_VERSION) {
                      bat 'pip install -r requirements.txt'
                      bat 'pip install jinja2'
                    }
                  }
                  withVenv{
                    dir('test/host') {
                      bat 'pytest -s'
                    }
                  }
                }
              }
            }
          } // stages
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        }// Windows

        // Host app on Linux
        stage ('Linux x86_64 Host Build & Test') {
          agent {
            label 'linux&&x86_64'
            }
          stages {
            stage ('Build') {
              steps {
                runningOn(env.NODE_NAME)
                // build
                dir("lib_audio_dsp") {
                  checkout scm
                }
                dir('lib_audio_dsp/host') {
                  // Enable the XTC tools for xSCOPE
                  withTools(params.TOOLS_VERSION) {
                    sh 'cmake -B build -DTESTING=ON && cd build && make -j4'
                  }
                }
              }
            }
            stage ('Test') {
              steps {
                dir("lib_audio_dsp") {
                  createVenv("requirements.txt")
                  withVenv{
                    // Enable the XTC tools for xSCOPE
                    withTools(params.TOOLS_VERSION) {
                      sh 'pip install -r requirements.txt'
                      sh 'pip install jinja2'
                    }
                  }
                  withVenv{
                    dir('test/host') {
                      sh 'pytest -s'
                    }
                  }
                }
              }
            }
          } // stages
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Linux x86_64

        // Host app on Max x86_64
        stage ('Mac x86_64 Host Build & Test') {
          agent {
            label 'macos&&x86_64'
          }
          stages {
            stage ('Build') {
              steps {
                runningOn(env.NODE_NAME)
                // build
                dir("lib_audio_dsp") {
                  checkout scm
                }
                dir('lib_audio_dsp/host') {
                  // Enable the XTC tools for xSCOPE
                  withTools(params.TOOLS_VERSION) {
                    sh 'cmake -B build -DTESTING=ON && cd build && make -j4'
                  }
                }
              }
            }
            // The stage for the tests has not been added, see https://xmosjira.atlassian.net/browse/LCD-294 for more details.
          } // stages
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Mac x86_64

        // Host app on Max arm64
        stage ('Mac arm64 Host Build & Test') {
          agent {
            label 'macos&&arm64'
          }
          stages {
            stage ('Build') {
              steps {
                runningOn(env.NODE_NAME)
                // build
                dir("lib_audio_dsp") {
                  checkout scm
                }
                dir('lib_audio_dsp/host') {
                  withTools(params.TOOLS_VERSION) {
                    sh 'cmake -B build -DTESTING=ON && cd build && make -j4'
                  }
                }
              }
            }
            // The stage for the tests has not been added, see https://xmosjira.atlassian.net/browse/LCD-294 for more details.
          } // stages
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Mac arm64
      } // parallel
    } // CI
  } // stages
} // pipeline
