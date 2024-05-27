// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef __control_host_h__
#define __control_host_h__

#include "device_control_shared.h"

#ifdef __cplusplus
extern "C" {
#endif

#if USE_I2C && __xcore__
#include "i2c.h"
#include <xccompat.h>
#endif

#if USE_SPI
typedef enum spi_mode_t {
  SPI_MODE_0, /**< SPI Mode 0 - Polarity = 0, Clock Edge = 1 */
  SPI_MODE_1, /**< SPI Mode 1 - Polarity = 0, Clock Edge = 0 */
  SPI_MODE_2, /**< SPI Mode 2 - Polarity = 1, Clock Edge = 0 */
  SPI_MODE_3, /**< SPI Mode 3 - Polarity = 1, Clock Edge = 1 */
} spi_mode_t;
#endif

/**
 * \defgroup device_control_host
 *
 * The public API for using the device control host library
 * @{
 */

#if USE_XSCOPE  || __DOXYGEN__
/** Initialize the xscope host interface
 *
 *  \param host_str    String containing the name of the xscope host. Eg. "localhost"
 *  \param port_str    String containing the port number of the xscope host
 *
 *  \returns           Whether the initialization was successful or not
 */
control_ret_t control_init_xscope(const char *host_str, const char *port_str);
/** Shutdown the xscope host interface
 *
 *  \returns           Whether the shutdown was successful or not
 */
control_ret_t control_cleanup_xscope(void);
#endif

#if (!USE_USB && !USE_I2C && !USE_SPI && !USE_XSCOPE)
#error "Please specify transport for device control using USE_xxx define in build file"
#error "Eg. -DUSE_I2C=1 or -DUSE_USB=1 or -DUSE_SPI=1 or -DUSE_XSCOPE=1"
#endif

#if USE_I2C && __xcore__
/** Checks to see that the version of control library in the device is the same as the host
 *
 *  \param version      Reference to control version variable that is set on this call
 *  \param i_i2c        The xC interface used for communication with the I2C library (only for xCore I2C host)
 *
 *  \returns            Whether the checking of control library version was successful or not
 */
control_ret_t control_query_version(control_version_t *version,
                                    CLIENT_INTERFACE(i2c_master_if, i_i2c));
#else
/** Checks to see that the version of control library in the device is the same as the host
 *
 *  \param version      Reference to control version variable that is set on this call
 *
 *  \returns            Whether the checking of control library version was successful or not
 */
control_ret_t control_query_version(control_version_t *version);
#endif

/** Request to write to controllable resource inside the device. The command consists of a resource ID,
 *  command and a byte payload of length payload_len.
 *
 *  \param resid        Resource ID. Indicates which resource the command is intended for
 *  \param cmd          Command code. Note that this will be in the range 0x80 to 0xFF
 *                      because bit 7 set indiciates a write command
 *  \param payload      Array of bytes which constitutes the data payload
 *  \param payload_len  Size of the payload in bytes
 *
 *  \returns            Whether the write to the device was successful or not
 */
control_ret_t
control_write_command(control_resid_t resid, control_cmd_t cmd,
#if USE_I2C && __xcore__
                      CLIENT_INTERFACE(i2c_master_if, i_i2c),
#endif
                      const uint8_t payload[], size_t payload_len);

/** Request to read from controllable resource inside the device. The command consists of a resource ID,
 *  command and a byte payload of length payload_len.
 *
 *  \param resid        Resource ID. Indicates which resource the command is intended for
 *  \param cmd          Command code. Note that this will be in the range 0x80 to 0xFF
 *                      because bit 7 set indiciates a write command
 *  \param payload      Array of bytes which constitutes the data payload
 *  \param payload_len  Size of the payload in bytes
 *
 *  \returns            Whether the read from the device was successful or not
 */
control_ret_t
control_read_command(control_resid_t resid, control_cmd_t cmd,
#if USE_I2C && __xcore__
                     CLIENT_INTERFACE(i2c_master_if, i_i2c),
#endif
                     uint8_t payload[], size_t payload_len);

#ifdef __cplusplus
}
#endif

/**@}*/

#endif // __control_host_h__
